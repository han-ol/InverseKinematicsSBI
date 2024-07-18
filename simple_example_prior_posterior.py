import numpy as np

from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = True


def f_(x, y):
    return x**3 + y**2


def manifold_density_reweight_desired_(x, y):
    return ((3*x**2)**2 + (y)**2)**(-1/2)


x_max = 2
x_min = -2
y_max = np.sqrt(1-(-2)**3)
y_min = -np.sqrt(1-(-2)**3)

general_max = np.maximum(y_max, x_max)
general_min = np.minimum(y_min, x_min)

n = 500
x = np.linspace(general_min, general_max, n)
y = np.linspace(general_min, general_max, n)
X, Y = np.meshgrid(x, y)
Z = manifold_density_reweight_desired_(X, Y)
Z = (Z - Z.min()) / (Z.max() - Z.min())
Z = Z*(2-Z)

mask = f_(X, Y)

for Zm, name in [(Z, 'prior'), (np.ma.masked_where((1.3 < mask) | (mask < 0.7), Z), 'posterior_trivial')]:
    plt.pcolormesh(X, Y, Zm, shading='gouraud', color='blue', cmap='Blues')
    plt.xlim(general_min, general_max)
    plt.ylim(general_min, general_max)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(r'$\theta_1$', loc='center', fontsize=18)
    plt.ylabel(r'$\theta_2$', loc='center', fontsize=18)
    plt.gca().set_aspect("equal", adjustable='box')
    plt.tight_layout()
    plt.savefig(f'{name}.png')
    plt.show()