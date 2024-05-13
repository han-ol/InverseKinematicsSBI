# Plotting functions for inverse kinematics problem solutionsfrom
import matplotlib.pyplot as plt

def plot_arms(coords, fig_name):
    plt.figure() 
    n_arms = coords.shape[0]
    for i in range(min(n_arms,1000)):
        plt.plot(coords[i,:,0], coords[i,:,1], ls='-')
    plt.gca().set_aspect('equal')
    plt.savefig(f'{fig_name}.pdf')

