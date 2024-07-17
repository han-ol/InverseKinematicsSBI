# Plotting functions for inverse kinematics problem solutionsfrom
import matplotlib.pyplot as plt


def plot_arms(coords, fig_name=None, density=None):
    if density is not None:
        density = (density - density.min()) / (density.max() - density.min())
        density = density * (2 - density)
    n_arms = coords.shape[0]
    for i in range(min(n_arms, 5000)):
        if density is not None:
            plt.plot(coords[i, :, 0], coords[i, :, 1], ls="-", color=plt.get_cmap('Blues')(density[i]))
        else:
            plt.plot(coords[i, :, 0], coords[i, :, 1], ls="-", color=plt.get_cmap('Blues')((i+1)/(min(n_arms, 1000)+1)))
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    if fig_name is not None:
        plt.savefig(fig_name)
    else:
        plt.show()
