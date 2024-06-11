from itertools import combinations, product
from matplotlib.patheffects import withStroke
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx

def draw_3d_layer(center, size, color, label):
    """Draw a 3D layer box with Plotly."""
    x, y, z = center
    dx, dy, dz = size

    # Create the vertices of the box
    vertices = [
        [x-dx/2, y-dy/2, z-dz/2],
        [x+dx/2, y-dy/2, z-dz/2],
        [x+dx/2, y+dy/2, z-dz/2],
        [x-dx/2, y+dy/2, z-dz/2],
        [x-dx/2, y-dy/2, z+dz/2],
        [x+dx/2, y-dy/2, z+dz/2],
        [x+dx/2, y+dy/2, z+dz/2],
        [x-dx/2, y+dy/2, z+dz/2]
    ]

    # Define the faces of the box
    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [1, 2, 6, 5],  # right
        [0, 3, 7, 4]   # left
    ]

    # Flatten the list of vertices for Plotly
    x = [vertex[0] for vertex in vertices]
    y = [vertex[1] for vertex in vertices]
    z = [vertex[2] for vertex in vertices]
    
    # Define the i, j, k indices for the faces
    i = [face[0] for face in faces for _ in range(2)]
    j = [face[1] for face in faces for _ in range(2)]
    k = [face[2] for face in faces for _ in range(2)]
    
    # Duplicate the indices for the second triangle of each face
    i += [face[2] for face in faces for _ in range(2)]
    j += [face[3] for face in faces for _ in range(2)]
    k += [face[0] for face in faces for _ in range(2)]

    # Create the mesh
    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=0.5,
        flatshading=True
    )

    # Create the label
    label = go.Scatter3d(
        x=[center[0]], y=[center[1]], z=[center[2]],
        text=[label],
        mode='text',
        textposition='middle center'
    )

    return mesh, label

def visualize_unet_3d_boxes():
    fig = go.Figure()

    layers = [
        {"center": [0, 0, 0], "size": [64, 64, 16], "color": "blue", "label": "Init Conv"},
        {"center": [0, -100, 0], "size": [32, 32, 32], "color": "green", "label": "Down1"},
        {"center": [0, -200, 0], "size": [16, 16, 64], "color": "green", "label": "Down2"},
        {"center": [0, -300, 0], "size": [8, 8, 128], "color": "green", "label": "Down3"},
        {"center": [0, -400, 0], "size": [4, 4, 256], "color": "purple", "label": "Bottleneck"},
        {"center": [0, -500, 0], "size": [8, 8, 128], "color": "orange", "label": "Up1"},
        {"center": [0, -600, 0], "size": [16, 16, 64], "color": "orange", "label": "Up2"},
        {"center": [0, -700, 0], "size": [32, 32, 32], "color": "orange", "label": "Up3"},
        {"center": [0, -800, 0], "size": [64, 64, 16], "color": "red", "label": "Output Conv"}
    ]

    for layer in layers:
        mesh, label = draw_3d_layer(**layer)
        fig.add_trace(mesh)
        fig.add_trace(label)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        width=800,
        height=700,
        margin=dict(r=20, l=10, b=10, t=10)
    )

    fig.show()

def draw_layer(center, size, label):
    x, y, z = center
    dx, dy, dz = size

    # Create the vertices of the box
    vertices = [
        [x-dx/2, y-dy/2, z-dz/2],
        [x+dx/2, y-dy/2, z-dz/2],
        [x+dx/2, y+dy/2, z-dz/2],
        [x-dx/2, y+dy/2, z-dz/2],
        [x-dx/2, y-dy/2, z+dz/2],
        [x+dx/2, y-dy/2, z+dz/2],
        [x+dx/2, y+dy/2, z+dz/2],
        [x-dx/2, y+dy/2, z+dz/2]
    ]

    # Define the faces of the box
    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [1, 2, 6, 5],  # right
        [0, 3, 7, 4]   # left
    ]

    # Flatten the list of vertices for Plotly
    x = [vertex[0] for vertex in vertices]
    y = [vertex[1] for vertex in vertices]
    z = [vertex[2] for vertex in vertices]
    
    # Define the i, j, k indices for the faces
    i = [face[0] for face in faces for _ in range(2)]
    j = [face[1] for face in faces for _ in range(2)]
    k = [face[2] for face in faces for _ in range(2)]
    
    # Duplicate the indices for the second triangle of each face
    i += [face[2] for face in faces for _ in range(2)]
    j += [face[3] for face in faces for _ in range(2)]
    k += [face[0] for face in faces for _ in range(2)]

    # Create the mesh
    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color="blue",
        opacity=0.5,
        flatshading=True
    )

    # Create the label
    label = go.Scatter3d(
        x=[center[0]], y=[center[1]], z=[center[2]],
        text=[label],
        mode='text',
        textposition='middle center'
    )

    return mesh, label

def draw_arrow(start, end):
    return go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode='lines',
        line=dict(color='black', width=2),
        marker=dict(size=4)
    )

def visualize_unet_diagram():
    fig = go.Figure()

    layers = [
        {"center": [0, 0, 0], "size": [64, 64, 16], "label": "Init Conv"},
        {"center": [0, -100, 0], "size": [32, 32, 32], "label": "Down1"},
        {"center": [0, -200, 0], "size": [16, 16, 64], "label": "Down2"},
        {"center": [0, -300, 0], "size": [8, 8, 128], "label": "Down3"},
        {"center": [0, -400, 0], "size": [4, 4, 256], "label": "Bottleneck"},
        {"center": [0, -500, 0], "size": [8, 8, 128], "label": "Up1"},
        {"center": [0, -600, 0], "size": [16, 16, 64], "label": "Up2"},
        {"center": [0, -700, 0], "size": [32, 32, 32], "label": "Up3"},
        {"center": [0, -800, 0], "size": [64, 64, 16], "label": "Output Conv"}
    ]

    for layer in layers:
        mesh, label = draw_layer(**layer)
        fig.add_trace(mesh)
        fig.add_trace(label)

    # Add arrows for connections
    arrows = [
        draw_arrow([0, 0, 0], [0, -100, 0]),
        draw_arrow([0, -100, 0], [0, -200, 0]),
        draw_arrow([0, -200, 0], [0, -300, 0]),
        draw_arrow([0, -300, 0], [0, -400, 0]),
        draw_arrow([0, -400, 0], [0, -500, 0]),
        draw_arrow([0, -500, 0], [0, -600, 0]),
        draw_arrow([0, -600, 0], [0, -700, 0]),
        draw_arrow([0, -700, 0], [0, -800, 0])
    ]

    for arrow in arrows:
        fig.add_trace(arrow)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        width=800,
        height=700,
        margin=dict(r=20, l=10, b=10, t=10)
    )

    fig.show()

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def add_box(ax, text, xy, boxstyle="round,pad=0.3", box_color="lightblue", text_color="black"):
    """Add a text box to the plot."""
    ax.text(
        xy[0], xy[1], text, ha="center", va="center",
        bbox=dict(boxstyle=boxstyle, facecolor=box_color, edgecolor="black"),
        color=text_color, fontsize=10, family="monospace"
    )

def create_robokhutt_unet_diagram():
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(-12, 12)
    ax.set_ylim(-15, 15)
    ax.axis('off')

    # Coordinates for the layers
    coords = {
        "Input Image\n512x128x3": (-6, 13),
        "Initial Conv Block\n512x128x64": (-6, 11),
        "Downsample Layer 1\n256x64x128": (-6, 9),
        "Downsample Layer 2\n128x32x256": (-6, 7),
        "Downsample Layer 3\n64x16x512": (-6, 5),
        "Bottleneck\n512x512": (0, 3),
        "Embedding Layer 1\n1x512": (0, 1.5),
        "Embedding Layer 2\n1x256": (0, 0),
        "Embedding Layer 3\nn_cfeatx512": (0, -1.5),
        "Embedding Layer 4\nn_cfeatx256": (0, -3),
        "Upsample Layer 1\n128x32x512": (6, 5),
        "Upsample Layer 2\n256x64x256": (6, 7),
        "Upsample Layer 3\n512x128x128": (6, 9),
        "Upsample Layer 4\n512x128x64": (6, 11),
        "Output Conv Block\n512x128x3": (6, 13),
        "Output Image\n512x128x3": (6, 15)
    }

    # Define colors for each block
    colors = {
        "Input Image\n512x128x3": "lightblue",
        "Initial Conv Block\n512x128x64": "blue",
        "Downsample Layer 1\n256x64x128": "green",
        "Downsample Layer 2\n128x32x256": "green",
        "Downsample Layer 3\n64x16x512": "green",
        "Bottleneck\n512x512": "purple",
        "Embedding Layer 1\n1x512": "red",
        "Embedding Layer 2\n1x256": "red",
        "Embedding Layer 3\nn_cfeatx512": "red",
        "Embedding Layer 4\nn_cfeatx256": "red",
        "Upsample Layer 1\n128x32x512": "orange",
        "Upsample Layer 2\n256x64x256": "orange",
        "Upsample Layer 3\n512x128x128": "orange",
        "Upsample Layer 4\n512x128x64": "orange",
        "Output Conv Block\n512x128x3": "lightblue",
        "Output Image\n512x128x3": "lightblue"
    }

    # Add the boxes with corresponding colors
    for label, coord in coords.items():
        add_box(ax, label, coord, box_color=colors[label])

    # Add the arrows
    arrows = [
        ("Input Image\n512x128x3", "Initial Conv Block\n512x128x64"),
        ("Initial Conv Block\n512x128x64", "Downsample Layer 1\n256x64x128"),
        ("Downsample Layer 1\n256x64x128", "Downsample Layer 2\n128x32x256"),
        ("Downsample Layer 2\n128x32x256", "Downsample Layer 3\n64x16x512"),
        ("Downsample Layer 3\n64x16x512", "Bottleneck\n512x512"),
        ("Bottleneck\n512x512", "Embedding Layer 1\n1x512"),
        ("Embedding Layer 1\n1x512", "Embedding Layer 2\n1x256"),
        ("Embedding Layer 2\n1x256", "Embedding Layer 3\nn_cfeatx512"),
        ("Embedding Layer 3\nn_cfeatx512", "Embedding Layer 4\nn_cfeatx256"),
        ("Bottleneck\n512x512", "Upsample Layer 1\n128x32x512"),
        ("Upsample Layer 1\n128x32x512", "Upsample Layer 2\n256x64x256"),
        ("Upsample Layer 2\n256x64x256", "Upsample Layer 3\n512x128x128"),
        ("Upsample Layer 3\n512x128x128", "Upsample Layer 4\n512x128x64"),
        ("Upsample Layer 4\n512x128x64", "Output Conv Block\n512x128x3"),
        ("Output Conv Block\n512x128x3", "Output Image\n512x128x3")
    ]

    for start, end in arrows:
        ax.annotate('', xy=coords[end], xytext=coords[start],
                    arrowprops=dict(arrowstyle='->', color='black'))

    plt.show()

create_robokhutt_unet_diagram()
# visualize_unet_3d_boxes()
