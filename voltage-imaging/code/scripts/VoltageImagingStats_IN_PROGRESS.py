import sys
import numpy as np
import h5py
import os
import scipy.io
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'voltage-processing')))

import VoltageTraceStats as VT

from Packages.SimpleNeuronAnalysis.OrientationTuning.OTAnalysis import (
    FitResponse,
    OSIndex,
    FittingUtils,
)


def voltage_imaging_stats_analyis(
    cur_spike_event,
    cur_subthreshold_dFF,
    stim_tstamp,
    stim_tstamp_s,
    orient_angles_rad,
    t_test_alpha,
    anova_test_alpha,
):
    sys.dont_write_bytecode = True

    cur_spike_cnts = cnt_spike_in_time_stamp(cur_spike_event, stim_tstamp)
    cur_spike_rate = calculate_spike_rate_s(cur_spike_cnts, stim_tstamp_s)

    t_test_pvals, _ = stim_step_t_test(cur_spike_rate, test_steps=[1, 0])
    anova_test_result = stim_step_anova_oneway(cur_spike_rate, test_step=1)

    t_test_pass = holm_bonferrioni_comparison(t_test_pvals, t_test_alpha)
    anova_test_pass = anova_test_result.pvalue < anova_test_alpha

    suprathd_responses = calculate_suprathd_responses(cur_spike_rate)
    subthd_responses = calculate_subthd_responses(
        cur_subthreshold_dFF, stim_tstamp
    )

    roi_fr_responses = suprathd_responses

    fr_bounds = FittingUtils.est_double_gauss_fit_bounds(
        orient_angles_rad, roi_fr_responses
    )

    fr_double_gaussian_fit_obj = FitResponse.DoubleGaussian()
    fr_double_gaussian_fit_obj.fit(
        orient_angles_rad, roi_fr_responses, bounds=fr_bounds
    )

    fr_OSI = OSIndex.calculate_OSI(fr_double_gaussian_fit_obj)

    roi_subthd_responses = subthd_responses

    subthd_bounds = FittingUtils.est_double_gauss_fit_bounds(
        orient_angles_rad, roi_subthd_responses
    )

    subthd_double_gaussian_fit_obj = FitResponse.DoubleGaussian()
    subthd_double_gaussian_fit_obj.fit(
        orient_angles_rad, roi_subthd_responses, bounds=subthd_bounds
    )

    subthd_OSI = OSIndex.calculate_OSI(subthd_double_gaussian_fit_obj)

    return (
        t_test_pass,
        anova_test_pass,
        fr_OSI,
        subthd_OSI,
        fr_double_gaussian_fit_obj,
        subthd_double_gaussian_fit_obj,
        cur_spike_cnts,
        cur_spike_rate,
        t_test_pvals,
        anova_test_result,
        suprathd_responses,
        subthd_responses,
    )


# run this script as demo
if __name__ == "__main__":
    demo_src_file_path = "./DemoData/Voltage/demo_data.hdf5"

    stim_tstamp = None
    stim_tstamp_s = None
    orient_angles_rad = None
    roi_spike_events = None
    roi_subthd_dFFs = None

    with h5py.File(demo_src_file_path, "r") as hdf5_file:
        nof_rois = hdf5_file["nof_roi"][()]
        nof_frames = hdf5_file["nof_frames"][()]
        nof_trials = hdf5_file["nof_trials"][()]
        stim_tstamp = hdf5_file["stim_tstamp"][()]
        stim_tstamp_s = hdf5_file["stim_tstamp_s"][()]
        orient_angles_rad = hdf5_file["orient_angles_rad"][()]

        roi_spike_events = np.zeros((nof_rois, nof_trials, nof_frames))
        roi_subthd_dFFs = np.zeros((nof_rois, nof_trials, nof_frames))

        for i_roi in range(nof_rois):
            cur_roi_str = f"roi{i_roi}"
            roi_spike_events[i_roi, :, :] = hdf5_file[
                os.path.join(cur_roi_str, "cur_spike_event")
            ][()]
            roi_subthd_dFFs[i_roi, :, :] = hdf5_file[
                os.path.join(cur_roi_str, "cur_subthreshold_dFF")
            ][()]

    results = {"roi_data": []}

    for i_roi in range(nof_rois):
        cur_spike_event = roi_spike_events[i_roi, :, :]
        cur_subthreshold_dFF = roi_subthd_dFFs[i_roi, :, :]

        (
            t_test_pass,
            anova_test_pass,
            fr_OSI,
            subthd_OSI,
            fr_double_gaussian_fit_obj,
            subthd_double_gaussian_fit_obj,
            cur_spike_cnts,
            cur_spike_rate,
            t_test_pvals,
            anova_test_result,
            suprathd_responses,
            subthd_responses,
        ) = voltage_imaging_stats_analyis(
            cur_spike_event,
            cur_subthreshold_dFF,
            stim_tstamp,
            stim_tstamp_s,
            orient_angles_rad,
            0.05,
            0.05,
        )

        roi_result = {
            "subthreshold_dff": cur_subthreshold_dFF,
            "spike_events": cur_spike_event,
            "subthreshold_OSI": subthd_OSI,
            "fr_OSI": fr_OSI,
            "t_test_pass": t_test_pass,
            "anova_test_pass": anova_test_pass,
            "fr_double_gaussian_fit_params": fr_double_gaussian_fit_obj.p,
            "subthd_double_gaussian_fit_params": subthd_double_gaussian_fit_obj.p,
            "spike_counts": cur_spike_cnts,
            "spike_rates": cur_spike_rate,
            "t_test_p_values": t_test_pvals,
            "anova_test_result": anova_test_result.pvalue,
            "suprathreshold_responses": suprathd_responses,
            "subthreshold_responses": subthd_responses,
        }
        results["roi_data"].append(roi_result)
        print(f"ROI {i_roi}: Analysis complete.")

    scipy.io.savemat("voltage_imaging_results.mat", results)
    print("Results saved to voltage_imaging_results.mat")