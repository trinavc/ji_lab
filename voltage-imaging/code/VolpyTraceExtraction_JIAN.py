# NOTE: The VolPy data processing script was adopted from the example script from CaImAn
# REFERENCE: Kleinfeld, D. et al. CaImAn an open source tool for scalable calcium imaging data analysis. (2019) doi:10.7554/eLife.38173.001.

import sys
import cv2
import logging
import h5py
import numpy as np
from scipy import stats
import scipy.io as sio
from pathlib import Path
import os

import caiman as cm
from volparams import volparams
from volpy import VOLPY


from VoltageTraceOps import (
    verifySpikeSTD,
)

def volpy_trace_extraction(
    srcDataFilePath,
    srcROIFilePath,
    opts_dict,
    roi_idx,
    detrend_winsize,
    spike_winsize,
    threadshold,
):
    
    sys.dont_write_bytecode = True

    try:
        cv2.setNumThreads(0)
    except:
        pass

    try:
        if __IPYTHON__:
            # this is used for debugging purposes only. allows to reload classes
            # when changed
            get_ipython().magic('load_ext autoreload')
            get_ipython().magic('autoreload 2')
    except NameError:
        pass

    logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]" \
                    "[%(process)d] %(message)s",
                    level=logging.ERROR)

    srcROIs = cm.load(srcROIFilePath)
    if(len(srcROIs.shape) < 3):
        srcROIs = srcROIs.reshape((1,) + srcROIs.shape)
    srcROIs = (srcROIs > 0)

    _, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
    srcMvMmpFName = cm.save_memmap([srcDataFilePath], base_name='memmap_', dview = dview, order = 'C')

    ROIs = srcROIs   
    index = list(range(len(ROIs)))     

    opts_dict["fnames"] = srcMvMmpFName,
    opts_dict["ROIs"] = ROIs
    opts_dict["index"] = index   

    opts = volparams(params_dict=opts_dict)
    opts.change_params(params_dict=opts_dict) 

    vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
    vpy.fit(n_processes=n_processes, dview=dview) 

    dFF = vpy.estimates['dFF'][roi_idx]
    spikes_valid, _ = verifySpikeSTD(
        vpy.estimates['dFF'][roi_idx],  
        vpy.estimates['spikes'][roi_idx], 
        detrend_winsize, 
        spike_winsize, 
        threadshold, )
    
    return (dFF, spikes_valid)

def calculate_osi(responses):
    """Calculate orientation selectivity index"""
    # Assuming responses is array of responses for different orientations
    sum_resp = np.sum(responses * np.exp(2j * np.pi * np.arange(len(responses)) / len(responses)))
    return np.abs(sum_resp) / np.sum(responses)

def analyze_roi_data(dFF, spikes, subthreshold):
    """Calculate statistics for a single ROI"""
    # T-test: comparing response periods to baseline
    t_stat, t_pval = stats.ttest_1samp(dFF, 0)
    
    # ANOVA: comparing responses across different conditions
    f_stat, f_pval = stats.f_oneway(*[group for group in np.array_split(dFF, 8)])  # Assuming 8 conditions
    
    # Calculate OSIs
    fr_osi = calculate_osi(spikes)  # Firing rate OSI
    sub_osi = calculate_osi(subthreshold)  # Subthreshold OSI
    
    return {
        't_test': {'statistic': t_stat, 'pvalue': t_pval},
        'anova': {'statistic': f_stat, 'pvalue': f_pval},
        'fr_OSI': fr_osi,
        'subthreshold_OSI': sub_osi
    }

def process_hdf5_file(input_path, output_path):
    """Process HDF5 file and save results as .mat"""
    print(f"\nProcessing file: {input_path}")
    
    # Load HDF5 file
    with h5py.File(input_path, 'r') as f:
        print("Successfully opened HDF5 file")
        
        # Extract data for all ROIs
        results = {
            'spike_events': [],
            'subthreshold_dFF': [],
            't_test_stats': [],
            't_test_pvals': [],
            'anova_stats': [],
            'anova_pvals': [],
            'fr_OSI': [],
            'subthreshold_OSI': []
        }
        
        # Get number of ROIs
        n_rois = len(f['estimates/dFF'])
        print(f"Found {n_rois} ROIs to process")
        
        # Process each ROI
        for roi_idx in range(n_rois):
            print(f"\nProcessing ROI {roi_idx + 1}/{n_rois}")
            
            # Get data for this ROI
            print("  Loading ROI data...")
            dFF = f['estimates/dFF'][roi_idx]
            spikes = f['estimates/spikes'][roi_idx]
            subthreshold = f['selected_roi/t_sub'][roi_idx] if 'selected_roi/t_sub' in f else None
            
            # Calculate statistics
            print("  Calculating statistics...")
            stats_results = analyze_roi_data(dFF, spikes, subthreshold)
            
            # Store results
            print("  Storing results...")
            results['spike_events'].append(spikes)
            results['subthreshold_dFF'].append(subthreshold)
            results['t_test_stats'].append(stats_results['t_test']['statistic'])
            results['t_test_pvals'].append(stats_results['t_test']['pvalue'])
            results['anova_stats'].append(stats_results['anova']['statistic'])
            results['anova_pvals'].append(stats_results['anova']['pvalue'])
            results['fr_OSI'].append(stats_results['fr_OSI'])
            results['subthreshold_OSI'].append(stats_results['subthreshold_OSI'])
            print(f"  ROI {roi_idx + 1} complete")
    
    # Convert lists to numpy arrays
    print("\nConverting results to numpy arrays...")
    for key in results:
        results[key] = np.array(results[key])
    
    # Save as .mat file
    print(f"Saving results to: {output_path}")
    sio.savemat(output_path, results)
    print("Processing complete!")

if __name__ == "__main__":
    print("Starting statistical analysis...")
    
    # Set paths
    input_dir = "/Users/trinav/personal/research/voltage-imaging/data/outputs/step2"
    output_dir = "/Users/trinav/personal/research/voltage-imaging/data/outputs/step3"
    
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    print("Creating output directory...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all HDF5 files
    print("Looking for HDF5 files...")
    hdf5_files = list(Path(input_dir).glob('*.h5'))
    
    if not hdf5_files:
        print(f"Error: No HDF5 files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(hdf5_files)} HDF5 files to process")
    
    # Process each file
    for i, input_file in enumerate(hdf5_files, 1):
        print(f"\nProcessing file {i}/{len(hdf5_files)}: {input_file.name}")
        output_file = Path(output_dir) / f"{input_file.stem}_stats.mat"
        
        try:
            process_hdf5_file(str(input_file), str(output_file))
            print(f"Successfully processed {input_file.name}")
        except Exception as e:
            print(f"Error processing {input_file.name}: {e}")
            continue
    
    print("\nAll processing complete!")


                               
                      


