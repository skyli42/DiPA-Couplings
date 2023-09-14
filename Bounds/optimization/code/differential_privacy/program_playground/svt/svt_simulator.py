import numpy as np
import random
import scipy
import scipy.integrate

np.set_printoptions(suppress=True) # print without scientific notation

sensitivity = 1
epsilon = 0.1
c = 3

BOT = '⊥'
TOP = '⊤'

def superscript(n):
    return "".join(["⁰¹²³⁴⁵⁶⁷⁸⁹"[ord(c)-ord('0')] for c in str(n)]) 

def format_sequence(seq: str):
    if not seq:
        return seq

    out = ""

    cur_symbol = seq[0]
    cur_count = 1
    
    for i in range(1, len(seq)):
        if(seq[i] != cur_symbol):
            out += cur_symbol
            if cur_count > 1:
                out += superscript(cur_count)
            out += ' '
            cur_symbol = seq[i]
            cur_count = 1
        else:
            cur_count += 1
        
    out += cur_symbol
    if cur_count > 1:
        out += superscript(cur_count)

    return out

def laplace_cdf(x, b):
    if x <= 0:
        return (1/2) * np.exp(x / b)
    else:
        return 1 - (1/2) * np.exp(- x / b)
    
def laplace_pdf(x, b):
    return (1/(2 * b)) * np.exp(-np.abs(x)/b)

def f_D(z, b, database, threshold, output_sequence):
    prod = np.exp(0)
    for i in range(len(output_sequence)):
        query_val = database[i]
        if output_sequence[i] == Outputs.BOT:
            prod *= laplace_cdf(threshold - query_val + z, b)
    return prod

def g_D(z, b, database, threshold, output_sequence):
    prod = np.exp(0)
    for i in range(len(output_sequence)):
        query_val = database[i]
        if output_sequence[i] == Outputs.TOP:
            prod *= 1 - laplace_cdf(threshold - query_val + z, b)
    return prod

def prob_output(database, threshold, output_sequence):
    epsilon_1 = epsilon/2
    epsilon_2 = epsilon - epsilon_1

    alpha = sensitivity/epsilon_1

    beta = 2 * c * sensitivity / epsilon_2

    integrand = lambda z: (
        laplace_pdf(z, alpha) * f_D(z, beta, database, threshold, output_sequence) * g_D(z, beta, database, threshold, output_sequence)
    )

    return scipy.integrate.quad(integrand, -np.inf, np.inf)

"""
A differentially private algorithm parametrized by epsilon.

Args:
    database: database
    threshold: threshold

Returns:
    A string of containing BOT, TOP representing below and above threshold respectively.
"""
def above_threshold(database: np.array, threshold: int, print_flag = False) -> str:

    epsilon_1 = epsilon/2
    epsilon_2 = epsilon - epsilon_1

    count = 0

    alpha = sensitivity/epsilon_1
    rho = np.random.laplace(0, alpha)
    noised_threshold = threshold + rho

    beta = 2 * c * sensitivity / epsilon_2
    data_noise = np.random.laplace(0, beta, database.size)
    noised_database = database + data_noise

    if print_flag:
        print(f"ε := {epsilon}")
        print(f"c := {c} (maximum {Outputs.TOP} responses)")
        print(f"query sensitivity = {sensitivity}")
        print(f"alpha = {alpha} (scale parameter for threshold noise)")
        print(f"beta = {beta} (scale parameter for database noise)")

    # print(f"Noised database: {noised_database}")
    # print(f"Noised threshold: {noised_threshold}")

    output_string = ""

    for noised_val in noised_database:
        if noised_val >= noised_threshold:
            output_string += Outputs.TOP
            count += 1
            if count >= c:
                break
        else:
            output_string += Outputs.BOT

    if print_flag:
        print()
        print(f"above_threshold output (a): {format_sequence(output_string)}")

    return output_string

def exact_output(database: np.array, threshold: int, print_flag = False) -> str:

    count = 0
    output_string = ""

    for val in database:
        if val >= threshold:
            output_string += Outputs.TOP
            count += 1
            if count >= c:
                break
        else:
            output_string += Outputs.BOT

    # output_string = format_sequence(output_string)

    if print_flag:
        print(f"        correct output (b): {format_sequence(output_string)}")

    return output_string

lower = 10**6
upper = 10**6 + 4000
mult = 1
threshold = ((lower + upper)/2)*mult
database = np.arange(lower, upper)*mult
random.shuffle(database)

print(f"data: {database}")

print(f"database: randomly shuffled values from {lower*mult} to {upper*mult}")
print(f"threshold: {threshold}")
print()
out = above_threshold(database, threshold, print_flag = True)
correct = exact_output(database, threshold, print_flag = True)

out_prob = prob_output(database, threshold, out)[0]
correct_prob = prob_output(database, threshold, correct)[0]
print(f"\nProbabilities:\n")
print(f"P[A(D) = a] = {out_prob}")
print(f"P[A(D) = b] = {correct_prob}")
print(f"\nε-DP guarantee: {np.exp(-epsilon) * out_prob} <= P[A(D') = a] <= {np.exp(epsilon) * out_prob}")
print(f"                {np.exp(-epsilon) * correct_prob} <= P[A(D') = b] <= {np.exp(epsilon) * correct_prob}\n")

# output_tracker = {}
# trials = 200
# print(f"Running above_threshold {trials} times:")
# for i in range(trials):
#     out = above_threshold(database, threshold)
#     output_tracker[out] = output_tracker.get(out, 0) + 1
#     # out_prob = prob_output(database, threshold, out)[0]
#     print(out)
#
# dom = sorted(list(output_tracker.keys()))
# dom_formatted = [format_sequence(seq) for seq in sorted(list(output_tracker.keys()))]
# integrated = np.array([prob_output(database, threshold, out)[0] for out in dom])
# measured = np.array([output_tracker[out]/trials for out in dom])

# import matplotlib.pyplot as plt

# # Create the figure and axis
# fig, ax = plt.subplots(figsize=(8, 10))

# # Plotting the data with thinner points and interesting symbols
# ax.plot(integrated, dom_formatted, 'bx', markersize=6, alpha=1, label='Integrated')
# ax.plot(measured, dom_formatted, 'r+', markersize=6, alpha=1, label='Measured')

# # Setting up the y-axis ticks with increased spacing
# num_ticks = len(dom_formatted)
# ax.set_yticks(np.linspace(0, num_ticks - 1, num_ticks))
# ax.set_yticklabels(dom_formatted)

# # Adding labels and title
# ax.set_xlabel('Probability')
# ax.set_ylabel('SVT Outputs')
# ax.set_title('Integrated vs Measured Probabilities')

# # Adding a legend
# ax.legend()

# # Display the plot
# plt.tight_layout()
# plt.savefig("fig1.png")

# for out in :
#     out_prob = prob_output(database, threshold, out)[0]
#     print("{:<20} with integrated probability {:<12}".format(format_sequence(out), out_prob))
#     print("{:<20} with measured probability   {:<12}".format("", output_tracker[out]/trials))
#     print()

# print(f"\nCorrect output: ")
# out = exact_output(database, threshold)
# out_prob = prob_output(database, threshold, out)[0]
# print("{:<20} with output probability {:>12}".format(format_sequence(out), out_prob))