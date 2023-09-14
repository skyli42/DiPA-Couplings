import numpy as np


def laplace_pdf_without_x(mean, scale_inverse):
    return (scale_inverse / 2) * np.exp(- scale_inverse * np.abs(mean))


def laplace_pdf(x, mean, scale_inverse):
    return laplace_pdf_without_x(x - mean, scale_inverse)


def laplace_cdf(x, mean, scale_inverse):
    return 0.5 * (1 + np.sign(x - mean) * (1 - np.exp(- scale_inverse * np.abs(x - mean))))


def integral_of_laplace(mean, scale_inverse, l, u):
    return laplace_cdf(u, mean, scale_inverse) - laplace_cdf(l, mean, scale_inverse)
