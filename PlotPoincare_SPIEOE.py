import numpy as np
import matplotlib.pyplot as plt


def plot_points_on_sphere(points):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Kugeloberfläche (Radius 1) als Drahtgitter
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_s = np.outer(np.cos(u), np.sin(v))
    y_s = np.outer(np.sin(u), np.sin(v))
    z_s = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_s, y_s, z_s, color='teal', alpha=0.3)


    points = np.asarray(points)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', s=50)


    for i, (x, y, z) in enumerate(points):
        ax.text(x, y, z, str(i + 1), color='mediumblue', fontsize=12)


    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    return fig


# Stokes-Parameter
calculated = [
    [0.0,     0.0,    -1.0   ],
    [0.0,     0.0,     1.0   ],
    [-0.625,  0.6495, -0.433 ],
    [-0.25,   0.866,   0.433 ],
    [-0.0302, 0.171,  -0.9848],
    [-0.0302, 0.171,   0.9848],
    [-0.1736,-0.9848,  0.0   ],
]


fig = plot_points_on_sphere(calculated)
fig.savefig("Poincare_sphere_calculated.png", dpi=600, bbox_inches='tight')
plt.close(fig)