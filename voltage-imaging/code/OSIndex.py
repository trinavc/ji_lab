import numpy as np
from .FitResponse import DoubleGaussian


def calculate_OSI(double_gaussian_fit_obj : DoubleGaussian):

    double_gauss_params = double_gaussian_fit_obj.params

    A_pref = double_gauss_params[1]
    A_oppo = double_gauss_params[2]
    theta_pref = double_gauss_params[3]
    theta_oppo = theta_pref + np.pi
    if A_oppo > A_pref:
        tmp = theta_oppo
        theta_oppo = theta_pref
        theta_pref = tmp

    theta_ortho = np.mod(theta_pref, np.pi) + np.pi/2

    R_ortho = double_gaussian_fit_obj.apply(theta_ortho)
    R_pref = double_gaussian_fit_obj.apply(theta_pref)

    OSI = (R_pref - R_ortho)/(R_pref + R_ortho)

    return OSI



def calculate_DSI(double_gaussian_fit_obj : DoubleGaussian):

    double_gauss_params = double_gaussian_fit_obj.params

    A_pref = double_gauss_params[1]
    A_oppo = double_gauss_params[2]
    theta_pref = double_gauss_params[3]
    theta_oppo = theta_pref + np.pi
    if A_oppo > A_pref:
        tmp = theta_oppo
        theta_oppo = theta_pref
        theta_pref = tmp

    R_pref = double_gaussian_fit_obj.apply(theta_pref)
    R_oppo = double_gaussian_fit_obj.apply(theta_oppo)

    DSI = (R_pref - R_oppo)/(R_pref + R_oppo)

    return DSI
    


    
