# Complete Revised VolPy Data Processing Script
import sys
import cv2
import logging
import os
import glob
import h5py
import numpy as np
import caiman as cm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'voltage-processing')))
                                             
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY
from VoltageTraceOps import verifySpikeSTD
import tifffile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_tif_stack(tif_path):
    """ Manually load the multi-frame TIFF file to ensure VolPy reads all frames correctly. """
    logging.info(f"Loading TIFF file: {tif_path}")
    
    # Read the TIFF stack
    tif_stack = tifffile.imread(tif_path)  # Expected shape: (frames, height, width)

    if len(tif_stack.shape) == 2:
        logging.warning("TIFF appears to be a single frame. Converting to multi-frame artificially.")
        tif_stack = np.expand_dims(tif_stack, axis=0)  # Add frame dimension

    logging.info(f"✅ TIFF Loaded: {tif_stack.shape}")  # Should be (T, H, W)
    
    # Convert to CaImAn format
    srcROIs = np.array(tif_stack, dtype=np.float32)  # Ensure dtype is correct
    if len(srcROIs.shape) == 3:
        srcROIs = srcROIs[:, np.newaxis, :, :]  # Reshape to (T, 1, H, W) for CaImAn

    logging.info(f"✅ Reshaped for CaImAn: {srcROIs.shape}")
    return srcROIs


def extract_metadata_from_tif(tif_path):
    logging.info(f"Extracting metadata from {tif_path}")
    with TiffFile(tif_path) as tif:
        nof_frames = len(tif.pages)
        metadata = tif.pages[0].tags['ImageDescription'].value if 'ImageDescription' in tif.pages[0].tags else ''
        nof_trials = int(metadata.split('nof_trials=')[1].split()[0]) if 'nof_trials=' in metadata else 1
        orient_angles_rad = np.array([float(a) for a in metadata.split('angles=')[1].split()]) if 'angles=' in metadata else np.linspace(0, np.pi, 8)
        stim_tstamp = np.fromstring(metadata.split('stim_tstamp=')[1], sep=',').reshape((8, 3, 2)) if 'stim_tstamp=' in metadata else np.zeros((8, 3, 2))
        stim_tstamp_s = np.fromstring(metadata.split('stim_tstamp_s=')[1], sep=',').reshape((8, 3, 2)) if 'stim_tstamp_s=' in metadata else np.zeros((8, 3, 2))
    logging.info(f"Extracted metadata: frames={nof_frames}, trials={nof_trials}")
    return nof_frames, nof_trials, orient_angles_rad, stim_tstamp, stim_tstamp_s

def save_to_hdf5(h5file, data_dict):
    logging.info("Saving data to HDF5 file")
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            h5file.create_dataset(key, data=value)
        elif isinstance(value, (int, float, str)):
            h5file.attrs[key] = value
        elif isinstance(value, dict): # Save nested dictionaries
            group = h5file.create_group(key)
            save_to_hdf5(group, value)
        else:
            h5file.attrs[key] = str(value) # Convert to string if other type
            logging.warning(f"Could not save {key} as a dataset, saving as attribute string")

def volpy_trace_extraction(srcDataFilePath, srcROIFilePath, opts_dict, roi_idx, detrend_winsize, spike_winsize, threshold, savePath=None, threshold_method = 'adaptive_threshold'):
    logging.info("Starting VolPy trace extraction")

    srcROIs = load_tif_stack(srcROIFilePath)
    if len(srcROIs.shape) < 3:
        srcROIs = srcROIs.reshape((1,) + srcROIs.shape)
    srcROIs = (srcROIs > 0)

    imaging_data = load_tif_stack(srcDataFilePath)
    if len(imaging_data.shape) < 3:
        imaging_data = imaging_data.reshape((1,) + imaging_data.shape)

    _, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=1, single_thread=True)

    srcMvMmpFName = cm.save_memmap([imaging_data[:5000]], base_name='memmap_', dview=dview, order='C')

    #ROIs = srcROIs
    #index = list(range(len(ROIs)))

    N = 2  # Change this number to select how many ROIs to process
    ROIs = srcROIs[:N]  # Select only the first N ROIs
    index = list(range(len(ROIs)))  # Update index list accordingly
    
    opts_dict["fnames"] = srcMvMmpFName
    opts_dict["ROIs"] = ROIs
    opts_dict["index"] = index

    opts = volparams(params_dict=opts_dict)
    opts.change_params(params_dict=opts_dict)
    # Ensure VolPy has a valid context_size
    logging.info(f"🔍 Checking VolPy default parameters: context_size = {opts.volspike.get('context_size', 'Not Defined')}")

    if np.sum(ROIs) == 0:
        raise ValueError("❌ Error: ROI mask is empty! Check Masks.tif.")
    logging.info(f"✅ ROI mask sum: {np.sum(ROIs)} (should be > 0)")



    vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
    vpy.fit(n_processes=n_processes, dview=dview)

    results = {}

    for i, roi in enumerate(ROIs):
        logging.info(f"Processing ROI {i+1}/{len(ROIs)}")
        dFF = vpy.estimates['dFF'][i]
        spikes = vpy.estimates['spikes'][i]

        spikes_valid, spikes_invalid = verifySpikeSTD(dFF, spikes, detrend_winsize, spike_winsize, threshold)
        results[f'roi{i}/cur_spike_event'] = spikes_valid
        results[f'roi{i}/cur_subthreshold_dFF'] = dFF

    results['opts'] = opts_dict
    results['vpy_estimates'] = vpy.estimates
    results['ROIs'] = ROIs

    if savePath:
        save_name = f'volpy{os.path.split(srcMvMmpFName)[1][:-5]}_{threshold_method}.hdf5'
        with h5py.File(os.path.join(savePath, save_name), 'w') as f:
            save_to_hdf5(f, results)

    logging.info("VolPy trace extraction completed successfully")
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

    return results

if __name__ == "__main__":
    opts_dict = {'fr': 1000/1.7}
    roi_idx = 0
    detrend_winsize = 100
    spike_winsize = 20
    threshold = 2.5

    filePath = r"/Users/trinav/personal/ji_lab/voltage-imaging/data"

    srcDataFilePath = os.path.join(filePath, 'Combined.tif')
    srcROIFilePath = os.path.join(filePath, "Masks.tif")

    logging.info(f"Processing data from {srcDataFilePath} and {srcROIFilePath}")

    volpy_trace_extraction(srcDataFilePath, srcROIFilePath, opts_dict, roi_idx, detrend_winsize, spike_winsize, threshold, savePath=filePath)

    logging.info("Processing complete.")

##    headpath = r"/home/jilab/anna_FACED_data/"
##    filePathArray = [r"AY124/FOV1", r"AY124/FOV2", r"AY124/FOV3", r"AY124/FOV4", r"AY124/FOV5", r"AY125/FOV1", r"AY125/FOV2"]
##    for n in list(range(len(filePathArray))):
##        filePath = os.path.join(headpath, filePathArray[n])
##        print(filePath)
##       srcDataFilePath = os.path.join(filePath, 'Combined.tif')
##        srcROIFilePath = os.path.join(filePath, "Masks.tif")
##        volpy_trace_extraction(srcDataFilePath, srcROIFilePath, opts_dict, roi_idx, detrend_winsize, spike_winsize, threshold, savePath=filePath)
##        print("Next Processing: ") 
    print("Finished...")