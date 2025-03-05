# NOTE: The VolPy data processing script was adopted from the example script from CaImAn
# REFERENCE: Kleinfeld, D. et al. CaImAn an open source tool for scalable calcium imaging data analysis. (2019) doi:10.7554/eLife.38173.001.

import sys
import cv2
import logging

import caiman as cm
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY


from Packages.SimpleNeuronAnalysis.NeuralActivities.VoltageTraceOps import (
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
    vpy.estimates['ROIs'] = ROIs
    spikes_valid, _ = verifySpikeSTD(
        vpy.estimates['dFF'][roi_idx],  
        vpy.estimates['spikes'][roi_idx], 
        detrend_winsize, 
        spike_winsize, 
        threadshold, )

    save_name = f'volpy{os.path.split(srcMvMmpFName)[1][:-5]}_{threshold_method}.hdf5'
    with h5py.File(os.path.join(savePath, save_name), 'w') as hf:
        recursively_save_dict_contents_to_group(hf, 'estimates' vpy.estimates)
        recursively_save_dict_contents_to_group(hf, 'opts' vpy.params)
    with h5py.File(os.path.join(savePath, save_name), 'w') as hf:
        hf.create_dataset('estimates', data=vpy.estimates)
        hf.create_dataset('opts', data = vpy.params)

    cm.stop_server(dview = dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
        

    return (dFF, spikes_valid)


# run this script as demo script
if __name__ == "__main__":
    print()
    print("Usage:")
    print(r" * Set srcDataFilePath as the file path of the motion registered image sequence")
    print(r" * Set srcROIFilePath as the file path of the ROI stack (ROIs need to be saved as mask images)")
    print(r"NOTE: Please check the following VolPy notebook on how to use VolPy:")
    print(r"URL: https://github.com/flatironinstitute/CaImAn/blob/main/demos/notebooks/demo_pipeline_voltage_imaging.ipynb")
    print('Processing...')
    opts_dict = {'fr':1000/1.7}
    roi_idx = 0
    detrend_winsize = 100
    spike_winsize = 20
    threadshold = 2.5

    headpath = r"/home/jilab/anna_FACED_data/"
    filepathArray = [r"AY124/FOV1", r"AY124/FOV2", r"AY124/FOV3", r"AY124/FOV4", r"AY124/FOV5", r"AY125/FOV1", f"AY125/FOV2"]
    for n in list(range(len(filePathArray))):
        filePath = os.path.join(headpath, filePathArray[n])
        print(filePath)
        srcDataFilePath = os.path.join(filePath, 'Combined.tif')
        srcROIFilePath = os.path.join(filePath, "Masks.tif")
        volpy_trace_extraction(filePath, srcDataFilePath, srcROIFilePath, opts_dict, roi_idx, detrend_winsize, spike_winsize, threadshold)
        print("Next Processing: ")
    print("Finished...")


                               
                      


