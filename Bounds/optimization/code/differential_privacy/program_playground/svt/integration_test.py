import scipy 
import scipy.integrate
import numpy as np

def laplace_pdf(mean, scale):
    return (1 / (2 * scale)) * np.exp(-np.abs(mean) / scale)

print(scipy.integrate.quad(lambda x : laplace_pdf(x, 1), -np.inf, np.inf))