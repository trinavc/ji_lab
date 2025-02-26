# Complete Revised VolPy Data Processing Script with Dynamic Array Reshaping and Logging
import sys
import cv2
import logging
import os
import glob
import h5py
import numpy as np
import scipy.io
import caiman as cm
from volparams import volparams
from volpy import VOLPY
from VoltageTraceOps import verifySpikeSTD
from tifffile import TiffFile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        else:
            h5file.attrs[key] = value
            
def reshape_data(data, target_shape):
    logging.info(f"Reshaping data from shape {data.shape} to {target_shape}")
    data = np.array(data)
    padded_data = np.zeros(target_shape)
    padded_data.flat[:data.size] = data.flat[:]
    return padded_data if data.size < np.prod(target_shape) else data.reshape(target_shape)

def volpy_trace_extraction(srcDataFilePath, srcROIFilePath, opts_dict, detrend_winsize, spike_winsize, threshold, savePath=None):
    logging.info("Starting VolPy trace extraction")
    nof_frames, nof_trials, orient_angles_rad, stim_tstamp, stim_tstamp_s = extract_metadata_from_tif(srcDataFilePath)
    srcROIs = cm.load(srcROIFilePath)
    _, dview, _ = cm.cluster.setup_cluster(backend='local')
    srcMvMmpFName = cm.save_memmap([srcDataFilePath], base_name='memmap_', dview=dview, order='C')

    results = {'nof_frames': nof_frames, 'nof_roi': len(srcROIs), 'nof_trials': nof_trials, 'orient_angles_rad': orient_angles_rad, 'stim_tstamp': stim_tstamp, 'stim_tstamp_s': stim_tstamp_s}
    vpy = VOLPY(srcMvMmpFName, srcROIs, opts_dict, dview)

    target_shape = (nof_trials, nof_frames)
    for i, roi in enumerate(srcROIs):
        logging.info(f"Processing ROI {i+1}/{len(srcROIs)}")
        dFF = reshape_data(vpy.estimates['dFF'][i], target_shape)
        spikes = reshape_data(vpy.estimates['spikes'][i], target_shape)
        spikes_valid, spikes_invalid = verifySpikeSTD(dFF, spikes, detrend_winsize, spike_winsize, threshold)
        results[f'roi{i}/cur_spike_event'] = spikes_valid
        results[f'roi{i}/cur_subthreshold_dFF'] = dFF

    if savePath:
        with h5py.File(os.path.join(savePath, 'output_data.hdf5'), 'w') as f:
            save_to_hdf5(f, results)
    logging.info("VolPy trace extraction completed successfully")
    cm.stop_cluster(dview=dview)
    return results

if __name__ == "__main__":
    volpy_trace_extraction("/Users/trinav/personal/research/voltage-imaging/data/Combined.tif", "/Users/trinav/personal/research/voltage-imaging/data/Masks.tif", {'fr': 1000/1.71}, 150, 30, 2.0, savePath="path/to/save/output")
