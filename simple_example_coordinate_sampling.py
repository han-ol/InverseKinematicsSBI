import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rc('text.latex', preamble=r'\usepackage{bm}')


def x_param_curve(t):
    return np.stack((t, np.sqrt(1-t**3)), axis=1)


def x_param_derivative(t):
    return np.stack((np.ones_like(t), -3/2*t**2/np.sqrt(1-t**3)), axis=1)


def manifold_density_reweight_desired(xy):
    return ((3*xy[:, 0]**2)**2 + (xy[:, 1])**2)**(-1/2)


def manifold_density_reweight_actual_x_param(xy):
    return (np.ones_like(xy[:, 0])**2 + (3/2*xy[:, 0]**2/np.sqrt(1-xy[:, 0]**3))**2)**(-1/2)


def manifold_density_reweight_actual_y_param(xy):
    return ((2/3 * xy[:, 1]/((1-xy[:, 1]**2)**(2))**(1/3))**2 + np.ones_like(xy[:, 1])**2)**(-1/2)


def length_param_curve(min_val=-1.5, max_val=1, n=100):
    t_actual = np.linspace(min_val, max_val, n + 2)[1:-1]
    norm_derivative_t_actual = np.linalg.norm(x_param_derivative(t_actual), axis=1)
    s_actual = np.cumsum(norm_derivative_t_actual)/np.sum(norm_derivative_t_actual)
    s_wanted = np.linspace(0, 1, n)
    t_wanted = np.interp(s_wanted, s_actual, t_actual)
    xy_wanted = x_param_curve(t_wanted)
    return xy_wanted


xy = length_param_curve(min_val=-2, n=50000)
x_max = np.max(xy[:, 0])
x_min = np.min(-xy[:, 0])
y_max = np.max(xy[:, 1])
y_min = np.min(-xy[:, 1])
density_funcs = {
    r"just manifold": None,
    r"manifold prior density": lambda a: np.exp(-1/2*np.linalg.norm(a, axis=1)**2),
    r"manifold actual density": lambda a: np.exp(-1/2*np.linalg.norm(a, axis=1)**2) * manifold_density_reweight_desired(a),
    r"manifold proj theta_1": lambda a: np.exp(-1/2*a[:, 0]**2) * manifold_density_reweight_actual_x_param(a),
    r"manifold proj theta_2": lambda a: np.exp(-1/2*a[:, 1]**2) * manifold_density_reweight_actual_y_param(a),
    r"manifold proj theta_12": lambda a: np.exp(-1/2*a[:, 1]**2) * (manifold_density_reweight_actual_x_param(a) + manifold_density_reweight_actual_y_param(a))
}
general_max = np.maximum(y_max, x_max)
general_min = np.minimum(y_min, x_min)
for name, density_func in density_funcs.items():
    if density_func is None:
        plt.scatter(xy[:,0], xy[:,1], s=1, color=plt.get_cmap('Blues')(0.99))
        plt.scatter(xy[:,0], -xy[:,1], s=1, color=plt.get_cmap('Blues')(0.99))
    else:
        density = density_func(xy)
        density = (density - density.min())/(density.max() - density.min())
        density = density*(2-density)
        plt.scatter(xy[:, 0], xy[:, 1], s=1, c=density, cmap='Blues')
        plt.scatter(xy[:, 0], -xy[:, 1], s=1, c=density, cmap='Blues')
    plt.xlim(general_min, general_max)
    plt.ylim(general_min, general_max)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(r'$\theta_1$', loc='center', fontsize=18)
    plt.ylabel(r'$\theta_2$', loc='center', fontsize=18)
    plt.gca().set_aspect("equal", adjustable='box')
    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.show()

