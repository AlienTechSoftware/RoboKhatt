import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, product

def draw_3d_layer(ax, center, size, color, label):
    """Draw a 3D layer box."""
    r = [-size[0] / 2, size[0] / 2]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            ax.plot3D(*zip(s + center, e + center), color=color)
    ax.text(center[0], center[1], center[2], label, color=color, fontsize=12, ha='center', va='center')

def visualize_unet():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    layers = [
        {"center": [0, 0, 0], "size": [64, 64, 16], "color": "blue", "label": "Init Conv"},
        {"center": [0, -80, 0], "size": [32, 32, 32], "color": "green", "label": "Down1"},
        {"center": [0, -160, 0], "size": [16, 16, 64], "color": "green", "label": "Down2"},
        {"center": [0, -240, 0], "size": [8, 8, 128], "color": "green", "label": "Down3"},
        {"center": [0, -320, 0], "size": [4, 4, 256], "color": "purple", "label": "Bottleneck"},
        {"center": [0, -400, 0], "size": [8, 8, 128], "color": "orange", "label": "Up1"},
        {"center": [0, -480, 0], "size": [16, 16, 64], "color": "orange", "label": "Up2"},
        {"center": [0, -560, 0], "size": [32, 32, 32], "color": "orange", "label": "Up3"},
        {"center": [0, -640, 0], "size": [64, 64, 16], "color": "red", "label": "Output Conv"}
    ]

    for layer in layers:
        draw_3d_layer(ax, **layer)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

visualize_unet()
