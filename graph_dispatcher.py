# graph_dispatcher.py

import numpy as np
import matplotlib.pyplot as plt
from rotorbloqueefunctions import *  # Make sure this is imported if used
from avidefunction import *
from variationgcouple import *
from matplotlib.patches import Rectangle

def plot_to_ax(ax, title):
    ax.clear()
    if title == "Induction à vide - Radiale":
        fig, axs, _ = av()
        copy_ax(ax, axs[0])  # Radiale
    
    elif title == "Induction à vide - Tangentielle":
        fig, axs, _ = av()
        copy_ax(ax, axs[1])  # Tangentielle

    elif title == "Induction à rotor bloqué - Radiale":
        fig, axs = rb()
        copy_ax(ax, axs[0])

    elif title == "Induction à rotor bloqué - Tangentielle":
        fig, axs = rb()
        copy_ax(ax, axs[1])

    elif title == "Courant dans les barres à g=1":
        fig, axs = rb()
        copy_ax(ax, axs[2])

    elif title == "Densité de courant dans une barre":
        fig, axs = rb()
        copy_ax(ax, axs[3])

    elif title == "Couple électromagnétique vs glissement":
        fig, axs = compute_and_plot_couple_data_on_ax()
        copy_ax(ax, axs[0])

    elif title == "Courant statorique vs glissement":
        fig, axs = compute_and_plot_couple_data_on_ax()
        copy_ax(ax, axs[1])

    elif title == "Résistance statorique en fonction du glissement":
        fig, axs = compute_and_plot_couple_data_on_ax()
        copy_ax(ax, axs[2])

    elif title == "inductance ramenée au rotor en fonction du glissement":
        fig, axs = compute_and_plot_couple_data_on_ax()
        copy_ax(ax, axs[3])

    elif title == "Résistance ramenée au rotor en fonction du glissement":
        fig, axs = compute_and_plot_couple_data_on_ax()
        copy_ax(ax, axs[4])

    else:
        x = np.linspace(0, 10, 200)
        y = {
            "Exponential": np.exp(-0.3 * x),
            "Tangent": np.tan(x),
            "Random Curve": np.random.rand(len(x))
        }.get(title, np.sin(x))
        colors = {
            "Sinus": "#f39c12",
            "Cosinus": "#9b59b6",
            "Exponential": "#27ae60",
            "Tangent": "#e74c3c",
            "Random Curve": "#3498db"
        }
        ax.plot(x, y, color=colors.get(title, "#ecf0f1"), linewidth=2)
        ax.set_title(title, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)

    ax.set_facecolor("white")
    ax.figure.patch.set_facecolor("white")
    ax.figure.tight_layout()

def copy_ax(target_ax, source_ax):
    from matplotlib.patches import Rectangle

    target_ax.clear()

    # Copy lines
    for line in source_ax.lines:
        target_ax.plot(
            line.get_xdata(),
            line.get_ydata(),
            label=line.get_label(),
            color=line.get_color()
        )

    # Rebuild bar plots from rectangle patches
    bars_x = []
    bars_height = []
    width = None
    colors = []
    edgecolors = []

    for patch in source_ax.patches:
        if isinstance(patch, Rectangle):
            x, y = patch.get_xy()
            h = patch.get_height()
            w = patch.get_width()
            bars_x.append(x)
            bars_height.append(h)
            colors.append(patch.get_facecolor())
            edgecolors.append(patch.get_edgecolor())
            width = w  # assume uniform width

    if bars_x and bars_height:
        for i in range(len(bars_x)):
            target_ax.bar(
                bars_x[i],
                bars_height[i],
                width=width,
                color=colors[i],
                edgecolor=edgecolors[i]
            )

    target_ax.set_title(source_ax.get_title())
    target_ax.set_xlabel(source_ax.get_xlabel())
    target_ax.set_ylabel(source_ax.get_ylabel())
    target_ax.grid(True)
