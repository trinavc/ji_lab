# NOTE: The VolPy data processing script was adopted from the example script from CaImAn
# REFERENCE: Kleinfeld, D. et al. CaImAn an open source tool for scalable calcium imaging data analysis. (2019) doi:10.7554/eLife.38173.001.

import sys
import cv2
import logging
import os
import glob
import h5py
import numpy as np
import scipy.io  # Add this import for saving .mat files

import caiman as cm
from volparams import volparams
from volpy import VOLPY
from VoltageTraceOps import verifySpikeSTD

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Save dictionary contents to HDF5 file recursively
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, int, float, bool)):
            h5file[path + '/' + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + '/' + key, item)
        else:
            raise ValueError(f'Cannot save {type(item)} type')

def volpy_trace_extraction(
    srcDataFilePath,
    srcROIFilePath,
    opts_dict,
    detrend_winsize,
    spike_winsize,
    thredshold,
    savePath=None
):
    print("Starting extraction...")
    sys.dont_write_bytecode = True

    try:
        cv2.setNumThreads(0)
    except:
        pass

    try:
        print(f"Loading ROIs from: {srcROIFilePath}")
        srcROIs = cm.load(srcROIFilePath)
        print(f"ROIs loaded successfully. Shape: {srcROIs.shape}")
    except Exception as e:
        print(f"Error loading ROIs: {e}")
        raise

    if(len(srcROIs.shape) < 3):
        srcROIs = srcROIs.reshape((1,) + srcROIs.shape)
        print(f"Reshaped ROIs to: {srcROIs.shape}")
    srcROIs = (srcROIs > 0)

    print("Setting up cluster...")
    try:
        _, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
        print(f"Cluster setup complete with {n_processes} processes")
    except Exception as e:
        print(f"Error setting up cluster: {e}")
        raise

    print(f"Loading and memory mapping data from: {srcDataFilePath}")
    try:
        srcMvMmpFName = cm.save_memmap([srcDataFilePath], base_name='memmap_', dview=dview, order='C')
        print(f"Data mapped to: {srcMvMmpFName}")
    except Exception as e:
        print(f"Error mapping data: {e}")
        raise

    results = []  # Initialize a list to store results for all ROIs

    for roi_idx in range(len(srcROIs)):
        print(f"Processing ROI {roi_idx}")
        dFF = vpy.estimates['dFF'][roi_idx]
        spikes_valid, spikes_invalid = verifySpikeSTD(
            vpy.estimates['dFF'][roi_idx],  
            vpy.estimates['spikes'][roi_idx], 
            detrend_winsize, 
            spike_winsize, 
            thredshold)
        
        # Store results for the current ROI
        results.append({
            'roi_index': roi_idx,
            'dFF': dFF,
            'spikes_valid': spikes_valid,
            'spikes_invalid': spikes_invalid,
        })

        # Save results if savePath is provided
        if savePath is not None:
            print(f"Saving results for ROI {roi_idx} to {savePath}")
            try:
                # Create directory if it doesn't exist
                os.makedirs(savePath, exist_ok=True)
                
                # Create HDF5 filename based on source data
                base_name = os.path.splitext(os.path.basename(srcMvMmpFName))[0]
                save_name = f'volpy_{base_name}_threshold.h5'
                save_path = os.path.join(savePath, save_name)
                
                # Save to HDF5 file
                with h5py.File(save_path, 'w') as hf:
                    recursively_save_dict_contents_to_group(hf, 'estimates', vpy.estimates)
                    recursively_save_dict_contents_to_group(hf, 'opts', vpy.params)
                    
                    # Also save the specific ROI results
                    hf.create_dataset(f'selected_roi_{roi_idx}/dFF', data=dFF)
                    hf.create_dataset(f'selected_roi_{roi_idx}/spikes_valid', data=spikes_valid)
                    hf.create_dataset(f'selected_roi_{roi_idx}/spikes_invalid', data=spikes_invalid)
                    hf.attrs['roi_index'] = roi_idx
                print(f"Results for ROI {roi_idx} saved successfully to {save_path}")
            except Exception as e:
                print(f"Error saving results for ROI {roi_idx}: {e}")
                raise

    # Save all results to a .mat file
    if savePath is not None:
        mat_save_path = os.path.join(savePath, 'output_statistics.mat')
        scipy.io.savemat(mat_save_path, {'results': results})
        print(f"All results saved successfully to {mat_save_path}")

    # Cleanup
    print("Cleaning up...")
    cm.stop_cluster(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
    
    print("Processing complete!")
    return results  # Return all results

# run this script as demo script
if __name__ == "__main__":
    print("Starting main script...")
    
    # Parameters
    print("Setting up parameters...")
    opts_dict = {'fr': 1000/1.71}
    detrend_winsize = 150
    spike_winsize = 30
    thredshold = 2.0

    # Input paths
    print("Setting up file paths...")
    srcDataFilePath = "/Users/trinav/personal/research/voltage-imaging/data/Combined.tif"
    srcROIFilePath = "/Users/trinav/personal/research/voltage-imaging/data/Masks.tif"
    
    # Create output directory structure
    print("Creating output directories...")
    base_save_path = "/Users/trinav/personal/research/voltage-imaging/data"
    save_path = os.path.join(base_save_path, "outputs", "step2")
    os.makedirs(save_path, exist_ok=True)
    
    print("Verifying input files exist...")
    if not os.path.exists(srcDataFilePath):
        raise FileNotFoundError(f"Combined.tif not found at: {srcDataFilePath}")
    if not os.path.exists(srcROIFilePath):
        raise FileNotFoundError(f"Masks.tif not found at: {srcROIFilePath}")
    
    # Run analysis
    print("Starting volpy_trace_extraction...")
    try:
        results = volpy_trace_extraction(
            srcDataFilePath,
            srcROIFilePath,
            opts_dict,
            detrend_winsize,
            spike_winsize,
            thredshold,
            savePath=save_path
        )
        print("Analysis complete!")
        print(f"Results saved in: {save_path}")
    except Exception as e:
        print(f"Error during processing: {e}")
        raise

                               
                      


