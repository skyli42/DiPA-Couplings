import numpy as np
import scipy.integrate
import scipy.stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from helpers.laplace import laplace_pdf, laplace_cdf

epsilon_0 = 1  # Replace with your desired value
epsilon_1 = 1  # Replace with your desired value

n = 10  # Number of iterations
j_domain = np.arange(1, 20)
z_domain = np.arange(1, n + 1)

ratios = np.zeros((len(j_domain), len(z_domain)))

for i, z in enumerate(z_domain):  # Vary z from 1 to n
    for j, j_val in enumerate(j_domain):
        f_1 = lambda x: laplace_pdf(x, 0, epsilon_0) * (laplace_cdf(x, z, epsilon_1) ** j_val) * (
                    1 - laplace_cdf(x, z, epsilon_1))
        f_2 = lambda x: laplace_pdf(x, 1, epsilon_0) * (laplace_cdf(x, z - 1, epsilon_1) ** j_val) * (
                    1 - laplace_cdf(x, z + 1, epsilon_1))

        p1, p1_err = scipy.integrate.quad(f_1, -np.inf, np.inf, epsrel=1e-30, epsabs=1e-30)
        p2, p2_err = scipy.integrate.quad(f_2, -np.inf, np.inf, epsrel=1e-30, epsabs=1e-30)
        ratio = max(p1 / p2, p2 / p1)
        ratio_err = ratio * np.sqrt((p1_err / p1) ** 2 + (p2_err / p2) ** 2)

        ratios[j, i] = ratio

        print("\nz = " + str(z) + ", j = " + str(j_val) + ":")
        print("p1: " + str(p1) + "+-" + str(p1_err))
        print("p2: " + str(p2) + "+-" + str(p2_err))
        print("p1 / p2: ", ratio, "+-" + str(ratio_err))

Z, J = np.meshgrid(z_domain, j_domain)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Z, J, ratios, cmap='viridis')

plt.show()
