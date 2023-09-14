from helpers.laplace import laplace_pdf_without_x, laplace_cdf, integral_of_laplace, laplace_pdf
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt

def thing1():
    # Computes probabilities on a segment with $j$ L-transitions in sequence.
    epsilon = 0.01
    epsilon_0 = 4.0 * epsilon
    epsilon_1 = (1.0 / 2.0) * epsilon

    n = 20
    ratios = []
    ratios_err = []
    discrete_domain = range(1, n)
    domain = np.linspace(1, n, 1000)

    for j in discrete_domain:
        # sequence that achieves the tightness bound:
        z = 500
        # X_1 = [0] + [z] * j
        # X_2 = [1] + [z - 1] * j

        f_1 = lambda x: laplace_pdf(x, 0, epsilon_0) * (laplace_cdf(x, z, epsilon_1) ** j)
        f_2 = lambda x: laplace_pdf(x, 1, epsilon_0) * (laplace_cdf(x, z - 1, epsilon_1) ** j)

        p1, p1_err = scipy.integrate.quad(f_1, -np.inf, np.inf, epsrel=1e-30, epsabs=1e-30)
        p2, p2_err = scipy.integrate.quad(f_2, -np.inf, np.inf, epsrel=1e-30, epsabs=1e-30)
        ratio = max(p1 / p2, p2 / p1)
        ratio_err = ratio * np.sqrt((p1_err / p1) ** 2 + (p2_err / p2) ** 2)

        ratios.append(ratio)
        ratios_err.append(ratio_err)

        print("\nj = " + str(j) + ":")
        print("p1: " + str(p1) + "+-" + str(p1_err))
        print("p2: " + str(p2) + "+-" + str(p2_err))
        print("p1 / p2: ", max(p1 / p2, p2 / p1), "+-" + str(ratio_err))

    # Plot the ratios against j

    s_l_bound = np.exp(2 * domain * epsilon_1)
    j_bound = np.array([np.exp(2 * epsilon_0) for _ in domain])
    s_n_bound = np.exp(domain * epsilon_1 + epsilon_0)
    print("s_n_bound: ", s_n_bound)

    # Plot the bounds against j
    plt.plot(domain, s_l_bound, color='r', label="$S^J$-bound: " + r"$\exp(2\epsilon_1 j)$")
    plt.plot(domain, j_bound, color='g', label="$S^L$ bound: " + r"$\exp(2\epsilon_0)$")
    plt.plot(domain, s_n_bound, color='y', label="$S^N$ bound: " + r"$\exp(\sum_i \epsilon_i)$")

    plt.xlabel("j")
    plt.ylabel("Ratios")
    plt.title("Scatter plot of ratios against j")

    # Make scatter plot more prominent
    plt.errorbar(discrete_domain, ratios, yerr=ratios_err, fmt='o', color='black', label="Ratios from counterexample")
    # scatter_plot = plt.scatter(discrete_domain, ratios, color='black', marker='o', label="Ratios")
    # scatter_plot.set_zorder(10)

    plt.legend()

    plt.show()


def thing2():
    # Computes probabilities on a segment with $j$ L-transitions in sequence.
    epsilon = 0.1
    epsilon_0 = 1.0 * epsilon
    epsilon_1 = (1.0 / 1.0) * epsilon

    z = 1
    n = 50
    ratios = []
    ratios_err = []
    domain = np.arange(1, n)

    for j in domain:
        # sequence that achieves the tightness bound:
        # X_1 = [0] + [z] * j
        # X_2 = [1] + [z - 1] * j

        f_1 = lambda x: laplace_pdf(x, 0, epsilon_0) * (laplace_cdf(x, z, epsilon_1) ** j) * (1 - laplace_cdf(x, z, epsilon_1))
        f_2 = lambda x: laplace_pdf(x, 1, epsilon_0) * (laplace_cdf(x, z - 1, epsilon_1) ** j) * (1 - laplace_cdf(x, z + 1, epsilon_1))

        p1, p1_err = scipy.integrate.quad(f_1, -np.inf, np.inf, epsrel=1e-30, epsabs=1e-30)
        p2, p2_err = scipy.integrate.quad(f_2, -np.inf, np.inf, epsrel=1e-30, epsabs=1e-30)
        ratio = max(p1 / p2, p2 / p1)
        ratio_err = ratio * np.sqrt((p1_err / p1) ** 2 + (p2_err / p2) ** 2)

        ratios.append(ratio)
        ratios_err.append(ratio_err)

        print("\nj = " + str(j) + ":")
        print("p1: " + str(p1) + "+-" + str(p1_err))
        print("p2: " + str(p2) + "+-" + str(p2_err))
        print("p1 / p2: ", max(p1 / p2, p2 / p1), "+-" + str(ratio_err))

    # Plot the ratios against j

    s_l_bound = np.array([np.exp(2 * epsilon_0 + 2 * epsilon_1) for j in domain])

    # Plot the bounds against j

    plt.plot(domain, s_l_bound, color='r', label="$S^J$-bound: " + r"$\exp(2\epsilon_0 + 2\epsilon_1)$")

    plt.xlabel("j")

    plt.ylabel("Ratios")
    plt.title("Scatter plot of ratios against j, with errors")

    # Make scatter plot more prominent

    plt.errorbar(domain, ratios, yerr=ratios_err, fmt='o', color='black', label="Ratios from counterexample")
    # scatter_plot = plt.scatter(discrete_domain, ratios, color='black', marker='o', label="Ratios")
    # scatter_plot.set_zorder(10)

    plt.legend()

    plt.show()


def thing3():
    import numpy as np
    import scipy.integrate
    import matplotlib.pyplot as plt

    epsilon_0 = 1  # Replace with your desired value
    epsilon_1 = 1  # Replace with your desired value

    n = 50  # Number of iterations
    ratios = []
    ratios_err = []
    j_domain = np.arange(1, 20)

    for z in range(1, n + 1):  # Vary z from 1 to 100
        ratio_list = []

        for j in j_domain:
            f_1 = lambda x: laplace_pdf(x, 0, epsilon_0) * (laplace_cdf(x, z, epsilon_1) ** j) * (
                        1 - laplace_cdf(x, z, epsilon_1))
            f_2 = lambda x: laplace_pdf(x, 1, epsilon_0) * (laplace_cdf(x, z - 1, epsilon_1) ** j) * (
                        1 - laplace_cdf(x, z + 1, epsilon_1))

            p1, p1_err = scipy.integrate.quad(f_1, -np.inf, np.inf, epsrel=1e-30, epsabs=1e-30)
            p2, p2_err = scipy.integrate.quad(f_2, -np.inf, np.inf, epsrel=1e-30, epsabs=1e-30)
            ratio = max(p1 / p2, p2 / p1)
            ratio_err = ratio * np.sqrt((p1_err / p1) ** 2 + (p2_err / p2) ** 2)

            ratio_list.append(ratio)

            print("\nz = " + str(z) + ", j = " + str(j) + ":")
            print("p1: " + str(p1) + "+-" + str(p1_err))
            print("p2: " + str(p2) + "+-" + str(p2_err))
            print("p1 / p2: ", ratio, "+-" + str(ratio_err))

        ratios.append(ratio_list)

    # Convert ratios to a 2D array
    ratios = np.array(ratios)

    # Plot the ratios as a scalar function of two variables (z and j)
    plt.imshow(ratios, origin='lower', cmap='viridis', extent=[1, n, 1, 100])
    plt.colorbar(label='Ratios')
    plt.xlabel('j')
    plt.ylabel('z')
    plt.title('Ratios as a Scalar Function of z and j')
    plt.show()


def test_snake_segment():
    """
    Find a tight bound for the output probabilities
    :return:
    """

thing2()