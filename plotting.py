__author__ = 'Mark'
import matplotlib.pyplot as plt
import numpy


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

    y_axis_means = []
    for dataset in y_axis:
        dataset_mean=[]
        for group_no in range(x_max - x_min + 1):
            group = dataset[group_no::x_max - x_min + 1]
            mean = sum(group) / len(group)
            dataset_mean.append(mean)
        y_axis_means.append(dataset_mean)


    # Plot mean
    for color, label, dataset_mean in zip(colors, labels, y_axis_means):
        ax.plot(x_axis[:x_max - x_min + 1], dataset_mean, color=color, linestyle='-', label='{} mean'.format(label))

    ax.legend(loc='lower right')
    ax.axis([x_min - 1, x_max + 1, 0, 1])
    plt.grid(True)
    # Add a table at the bottom of the axes
    the_table = plt.table(
        cellText=numpy.around(y_axis_means, decimals=2),
        rowLabels=labels,
        #rowColours=colors,
        colLabels=range(x_min, x_max+1),
        loc='bottom',
        bbox=[0.20, -0.6, 0.75, 0.3]
    )
    plt.subplots_adjust(bottom=0.4)

    plt.ylabel("Recognition rate")
    plt.xlabel("Number of training")

    plt.show()