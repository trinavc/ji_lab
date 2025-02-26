import numpy as np
from scipy import stats


def cnt_spike_in_time_stamp(spike_event, stim_tstamp):

    assert len(spike_event.shape) == 2
    
    nof_traces = spike_event.shape[0]
    trace_len = spike_event.shape[1]
    nof_orint = stim_tstamp.shape[0]
    nof_steps = stim_tstamp.shape[1]
    
    spike_cnts = np.zeros((nof_orint, nof_steps, nof_traces), dtype = float)
    
    for i_orint in range(nof_orint):
        for i_step in range(nof_steps):
            tstamp_start = stim_tstamp[i_orint, i_step, 0]
            tstamp_start = max(0, tstamp_start)
            tstamp_end = stim_tstamp[i_orint, i_step, 1]
            tstamp_end = min(trace_len, tstamp_end)
            if tstamp_start >= tstamp_end:
                continue
            tstamp_spike_event_mask = spike_event[:, tstamp_start:tstamp_end] > 0
            nof_spikes = np.sum(tstamp_spike_event_mask.astype(int), axis = -1)
            spike_cnts[i_orint, i_step, :] = nof_spikes
            
    return spike_cnts


def calculate_spike_rate_s(spike_cnts, stim_tstamp_s):
    nof_orint = spike_cnts.shape[0]
    nof_steps = spike_cnts.shape[1]
    nof_traces = spike_cnts.shape[2]
    
    spike_rate = np.zeros((nof_orint, nof_steps, nof_traces), dtype = float)
    
    for i_orint in range(nof_orint):
        for i_step in range(nof_steps):
            stamp_t_s = stim_tstamp_s[i_orint, i_step, :]
            duration_s = stamp_t_s[-1] - stamp_t_s[0]
            if duration_s == 0:
                continue
            spike_rate[i_orint, i_step, :] = spike_cnts[i_orint, i_step, :].astype(float) / duration_s

    return spike_rate


def calculate_trace_mean_in_time_stamp(src_traces, stim_tstamp):
    assert len(src_traces.shape) == 2
    
    nof_traces = src_traces.shape[0]
    trace_len = src_traces.shape[1]
    nof_orint = stim_tstamp.shape[0]
    nof_steps = stim_tstamp.shape[1]
    
    trace_means = np.zeros((nof_orint, nof_steps, nof_traces), dtype = float)
    
    for i_orint in range(nof_orint):
        for i_step in range(nof_steps):
            tstamp_start = stim_tstamp[i_orint, i_step, 0]
            tstamp_start = max(0, tstamp_start)
            tstamp_end = stim_tstamp[i_orint, i_step, 1]
            tstamp_end = min(trace_len, tstamp_end)
            if tstamp_start >= tstamp_end:
                continue
            cur_trace_mean = np.mean(src_traces[:, tstamp_start:tstamp_end], axis = -1)
            trace_means[i_orint, i_step, :] = cur_trace_mean
            
    return trace_means


def calculate_mean_trace_max_in_time_stamp(src_traces, stim_tstamp):
    assert len(src_traces.shape) == 2
    
    nof_traces = src_traces.shape[0]
    trace_len = src_traces.shape[1]
    nof_orint = stim_tstamp.shape[0]
    nof_steps = stim_tstamp.shape[1]

    mean_trace = np.mean(src_traces, axis = 0)
    mean_trace_maxs = np.zeros((nof_orint, nof_steps))
    
    for i_orint in range(nof_orint):
        for i_step in range(nof_steps):
            tstamp_start = stim_tstamp[i_orint, i_step, 0]
            tstamp_start = max(0, tstamp_start)
            tstamp_end = stim_tstamp[i_orint, i_step, 1]
            tstamp_end = min(trace_len, tstamp_end)
            if tstamp_start >= tstamp_end:
                continue
            mean_trace_maxs[i_orint, i_step] = np.max(mean_trace[tstamp_start:tstamp_end])
            
    return mean_trace_maxs

def stim_step_t_test(spike_rate, test_steps = [1,0]):
    alternative = "greater"
    
    nof_orints = spike_rate.shape[0]
    test_results = []
    pvalues = np.zeros((nof_orints,))
    for i_orient in range(nof_orints):
        a = spike_rate[i_orient, test_steps[0], :]
        b = spike_rate[i_orient, test_steps[1], :] 
        result = stats.ttest_1samp(a - b, popmean = 0, alternative = alternative)
        test_results.append(result)
        pvalues[i_orient] = result.pvalue
    return pvalues, test_results


def stim_step_anova_oneway(spike_rate, test_step = 1):
    nof_orints = spike_rate.shape[0]
    comp_group = []
    for i_orint in range(nof_orints):
        comp_group.append(spike_rate[i_orint, test_step, :])
    result = stats.f_oneway(*comp_group)
    return result


def calculate_suprathd_responses(cur_spike_rate, stim_step_idx = 1, baseline_step_idx = 0):
    responses = np.mean(cur_spike_rate[:,stim_step_idx,:], axis = -1) - np.mean(cur_spike_rate[:,baseline_step_idx,:], axis = None)
    responses[responses < 0] = 0
    return responses


def calculate_subthd_responses(subthreshold_dFF, stim_tstamp):
    subthd_dFF_means = calculate_trace_mean_in_time_stamp(subthreshold_dFF, stim_tstamp)
    mean_subthd_dFF_max = calculate_mean_trace_max_in_time_stamp(subthreshold_dFF, stim_tstamp)
    responses = mean_subthd_dFF_max[:,1] - np.mean(subthd_dFF_means[:,0,:], axis = -1)
    responses[responses < 0] = 0
    return responses


def holm_bonferrioni_comparison(pvals, alpha):
    pvals = np.sort(pvals)
    m = len(pvals)
    correct_coeffs = 1/(np.arange(1, m+1)[::-1])
    effective_alphas = alpha * correct_coeffs
    return np.sum(pvals <= effective_alphas) > 0 
