import numpy as np 
import matplotlib.pyplot as plt

Delta = 1
epsilon = 1

def algorithm(Q, T_x=1, T_y=2):

    output = ""

    scale_factor = epsilon/Delta

    x = T_x + np.random.laplace(0, scale_factor)
    y = T_y + np.random.laplace(0, scale_factor)

    i = 0

    while i < len(Q):
        in_x = Q[i] + np.random.laplace(0, scale_factor)
        in_y = Q[i] + np.random.laplace(0, scale_factor)
        i += 1
        if (in_x < x) or (in_y >= y):
            output += "⊥ "
        else:
            output += "⊤ "
            break 

    while i < len(Q):
        in_x = Q[i] + np.random.laplace(0, scale_factor)
        in_y = Q[i] + np.random.laplace(0, scale_factor)
        i += 1
        if (in_x >= x) or (in_y < y):
            output += "⊥ "
        else:
            output += "⊤ "
            break
    return output

def build_histogram(Q, num_samples, T_x, T_y):
    outputs = []
    for _ in range(num_samples):
        output = algorithm(Q, T_x, T_y)
        outputs.append(output)

    # Count the occurrences of each output
    unique_outputs, counts = np.unique(outputs, return_counts=True)

    # Plot the histogram
    plt.bar(unique_outputs, counts)
    plt.xlabel('Output')
    plt.ylabel('Count')
    plt.title('Histogram of Outputs')
    plt.xticks(rotation=45)
    

if __name__ == "__main__":
    Q = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    num_samples = 10000
    T_x = -1
    T_y = 1
    build_histogram(Q, num_samples, T_x, T_y)
    build_histogram(Q+np.ones(len(Q)), num_samples, T_x, T_y)
    plt.show()