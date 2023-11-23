import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def plot_basic_2d(x, y, title, x_label, y_label, legend_label, path):
    plt.plot(x, y, 'b', label=legend_label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(path)
    plt.clf()


def plot_log_2d(x, y, title, x_label, y_label, legend_label, path):
    plt.plot(x, y, 'b', label=legend_label)
    plt.yscale("log")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(path)
    plt.clf()


def plot_log_log_2d(x, y, title, x_label, y_label, legend_label, path):
    plt.plot(x, y, 'b', label=legend_label)
    plt.xscale("log")
    plt.yscale("log")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(path)
    plt.clf()


def plot_bar_with_labels(x_values, y_values, max_value, title, path):
    plt.figure(figsize=(24, 12))

    plt.bar(x_values, y_values, edgecolor='black')
    # plt.title(title)
    plt.ylim(0, max_value)

    plt.xticks(fontsize=32, rotation=90)  # Set font size for tick labels on x-axis
    plt.yticks(fontsize=32)  # Set font size for tick labels on y-axis

    # Adjust margins to reduce white space
    # plt.subplots_adjust(left=0.10, right=0.92, top=0.95, bottom=0.05)
    plt.tight_layout()

    plt.savefig(path)
    plt.clf()


def plot_heatmap(data):
    # Set the figure size (width, height) in inches
    fig = plt.figure(figsize=(5, 10))

    # Plot the heatmap with custom colors and annotations
    plt.imshow(data, vmin=0, vmax=100, extent=[0, 7, 0, 30])
    plt.colorbar()
    for i in range(30):
        for j in range(7):
            plt.text(j + 0.5, i + 0.5, str(data[i][j]),
                     ha='center', va='center', color='white')

    # Set x and y axis labels and tick positions
    x_labels = ["DE", "COMDE", "DERL", "EMDE", "SADE", "NMDE", "IDE"]
    y_labels = [
        'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9',
        'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19',
        'F20', 'F21', 'F22', 'F23', 'F24', 'F25', 'F26', 'F27', 'F28', 'F29', 'F30'
    ]
    plt.xticks(np.arange(len(x_labels)) + 0.5, x_labels, rotation=45)
    plt.yticks(np.arange(len(y_labels)) + 0.5, y_labels)

    plt.xticks(fontsize=12)  # Set font size for tick labels on x-axis
    plt.yticks(fontsize=14)  # Set font size for tick labels on y-axis

    plt.tight_layout()
    plt.show()


def plot_multiple_bar_with_labels(x_values, y_values, max_value, title, path):
    # y = np.arange(len(x_values))  # the label locations
    # height = 0.25  # the height of the bars
    # multiplier = 0
    #
    # fig, ax = plt.subplots(figsize=(12, 10))
    #
    # for attribute, measurement in y_values.items():
    #     offset = height * multiplier
    #     ax.barh(y + offset, measurement, height, label=attribute)
    #     multiplier += 1
    #
    # # Add some text for labels, title and custom y-axis tick labels, etc.
    # ax.set_xlabel('Length (mm)')
    # ax.set_title(title)
    # ax.set_yticks(y + height, x_values)
    # ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncols=1)
    # ax.set_xlim(0, max_value)
    #
    # plt.tight_layout()  # Ensures labels are not cut off
    # plt.savefig(path)
    # plt.clf()

    # Sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.exp(-x / 5) * np.cos(x)

    # Create a figure with a 2x2 grid of subplots
    plt.figure(figsize=(10, 6))

    # Plot on the first subplot
    plt.subplot(2, 2, 1)
    plt.plot(x, y1)
    plt.title('Subplot 1: sin(x)')

    # Plot on the second subplot
    plt.subplot(2, 2, 2)
    plt.plot(x, y2)
    plt.title('Subplot 2: cos(x)')

    # Plot on the third subplot
    plt.subplot(2, 2, 3)
    plt.plot(x, y3)
    plt.title('Subplot 3: exp(-x/5) * cos(x)')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the combined plot
    plt.show()
