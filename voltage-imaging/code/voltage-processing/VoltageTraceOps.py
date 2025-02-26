import numpy as np
from scipy import signal


def matchSpikeToPeak(trace, spikes, search_win_size = 3):
    assert(trace.ndim == 1)
    assert(spikes.ndim == 1)
    search_win_half_size = np.int64(2 * np.floor(search_win_size / 2))
    serach_center = search_win_half_size + 1
    search_win_size = 2*search_win_half_size + 1
    trace_roll_zero = np.zeros((search_win_size,) + trace.shape)
    last_idx = len(trace) - 1
    for i_step in range(search_win_half_size):
        trace_roll_zero[serach_center + i_step, 0:last_idx-i_step] = trace[i_step:last_idx]
        trace_roll_zero[serach_center - i_step, i_step:last_idx] = trace[0:last_idx-i_step]
    matched_spikes = np.argmax(trace_roll_zero[:, spikes], axis = 0) - serach_center + spikes

    return matched_spikes


def verifySpikeSTD(trace, spikes, detrend_win_size = 100, spike_win_size = 10, thredshold = 2):
    spikes = matchSpikeToPeak(trace, spikes)
    
    detrend_win_half_size = np.int64(np.floor(detrend_win_size/2))
    detrend_weights = np.linspace(-detrend_win_half_size, +detrend_win_half_size, 2*detrend_win_half_size+1, dtype = np.float32)
    detrend_weights = np.power(np.abs(detrend_weights), 0.5)
    detrend_weights = np.max(detrend_weights) - detrend_weights
    detrend_weights = detrend_weights / np.sum(detrend_weights)
    trend = np.convolve(trace, detrend_weights, 'same')
    trace_detrend = trace - trend
    
    spike_mask = np.zeros(trace.shape)
    spike_mask[spikes] = 1
    spike_mask = np.convolve(spike_mask, np.ones((np.int64(2*np.floor(spike_win_size/2) + 1),)), 'same')
    spike_mask = spike_mask > 0
    
    detrend_mean = np.mean(trace_detrend[np.logical_not(spike_mask)])
    detrend_std = np.std(trace_detrend[np.logical_not(spike_mask)])
    detrend_thredshold_val = detrend_mean + thredshold * detrend_std
    spikes_valid = spikes[trace_detrend[spikes] >= detrend_thredshold_val]
    spikes_invalid = spikes[trace_detrend[spikes] < detrend_thredshold_val]

    return (spikes_valid, spikes_invalid)


def bw_lp_filtering(order, cutoff, fs, src_traces):
    b, a = signal.butter(order, cutoff, "lp", fs = fs, output = "ba")
    dst_traces = signal.filtfilt(b, a, src_traces, method="gust", axis = -1)
    return dst_traces


def moving_avg(src_traces, win_size = 3):
    trace_len = src_traces.shape[-1]

    nof_avged_pnts = trace_len - win_size + 1

    avged_traces = np.zeros(src_traces.shape[:-1] + (nof_avged_pnts,), dtype = src_traces.dtype)
    for i_avgpnt in range(nof_avged_pnts):
        cur_win_start = i_avgpnt
        cur_win_end = i_avgpnt + win_size

        cur_avg_pnts = np.mean(src_traces[...,cur_win_start:cur_win_end], axis = -1)

        avged_traces[...,i_avgpnt] = cur_avg_pnts

    return avged_traces

