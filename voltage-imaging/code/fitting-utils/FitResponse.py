import scipy.optimize 
import numpy as np
from .DoubleGaussian import double_gaussian


class DoubleGaussian:
    def __init__(self, params = None):
        self.params = params

    def fit(self, theta, response, init_guess = None, bounds = (-np.inf, np.inf),**kwargs):
        fit_result = scipy.optimize.curve_fit(double_gaussian, 
                                              theta, 
                                              response, 
                                              p0 = init_guess,
                                              bounds = bounds,
                                              **kwargs)
        self.params = fit_result[0]
        return fit_result
 
    def apply(self, theta):
        fitted_response = double_gaussian(theta, *self.params)
        return fitted_response