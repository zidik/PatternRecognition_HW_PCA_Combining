__author__ = 'Mark'
import matplotlib.pyplot as plt


def plot_results(x_axis, y_axis, x_min, x_max, labels):
    try:
        y_axis[0][0]
    except IndexError:
        #Convert 1D list to 2D
        y_axis = [y_axis]

    colors = ('blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black')
    # Plot datapoints
    fig, ax = plt.subplots()

    for color, label, dataset in zip(colors, labels, y_axis):
        ax.plot(x_axis, dataset, color=color, marker='.', linestyle=' ', alpha=0.3, label='{} datapoints'.format(label))

    # Plot mean
    for color, label, dataset in zip(colors, labels, y_axis):
        plot_mean_y = []
        for group_no in range(x_max - x_min + 1):
            group = dataset[group_no::x_max - x_min + 1]
            mean = sum(group) / len(group)
            plot_mean_y.append(mean)
        ax.plot(x_axis[:x_max - x_min + 1], plot_mean_y, color=color, linestyle='-', label='{} mean'.format(label))

    ax.legend(loc='lower right')
    ax.axis([x_min - 1, x_max + 1, 0, 1])
    plt.grid(True)
    plt.show()