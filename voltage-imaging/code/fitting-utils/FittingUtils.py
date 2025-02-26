import numpy as np


def est_double_gauss_fit_init_guess(orint_angle_rad, reponses):
    nof_responses = len(reponses)
    max_response_idx = np.argmax(reponses)
    oppo_response_idx = nof_responses - 1 - max_response_idx
    if max_response_idx > nof_responses//2:
        tmp = max_response_idx
        max_response_idx = oppo_response_idx
        oppo_response_idx = max_response_idx
    init_guess = [0,
                  max(reponses[max_response_idx], 0),
                  max(reponses[oppo_response_idx], 0),
                  orint_angle_rad[max_response_idx],
                  np.pi/(nof_responses*2)]
    return init_guess


def est_double_gauss_fit_bounds(orint_angle_rad, reponses):
    max_response_ratio = 1
    bounds = (
        [np.min(reponses), 
         0, 
         0, 
         0, 
         0],
        [np.max(reponses), 
         max_response_ratio*(np.max(reponses) - np.min(reponses)), 
         max_response_ratio*(np.max(reponses) - np.min(reponses)), 
         np.pi, 
         np.pi/2]
    )
    return bounds