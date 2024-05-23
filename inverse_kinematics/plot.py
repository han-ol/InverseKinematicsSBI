# Plotting functions for inverse kinematics problem solutionsfrom
import matplotlib.pyplot as plt
import numpy as np


def plot_arms_2d(coords, fig_name=None):
    plt.figure() 
    n_arms = coords.shape[0]
    for i in range(min(n_arms,1000)):
        plt.plot(coords[i, :, 0], coords[i, :, 1], ls='-')
    plt.gca().set_aspect('equal')
    if fig_name is None:
        plt.show()
    else:
        plt.savefig(f'{fig_name}.pdf')


def plot_arms_3d(coords, fig_name=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    n_arms = coords.shape[0]
    for i in range(min(n_arms,1000)):
        plt.plot(coords[i, :, 0], coords[i, :, 1], coords[i, :, 2], ls='-')
    set_axes_equal(ax)
    if fig_name is None:
        plt.show()
    else:
        plt.savefig(f'{fig_name}.pdf')


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    From: https://stackoverflow.com/questions/13685386/how-to-set-the-equal-aspect-ratio-for-all-axes-x-y-z

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

