import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def getdimensions(data):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    c = data[:, 3]
    return x,y,z,c

def plot_4D(observations, predictions):
    x1, y1, z1, c1 = getdimensions(observations)
    x2, y2, z2, c2 = getdimensions(predictions)
    fig = plt.figure()
    fig.set_size_inches(20, 15)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('real')
    img1 = ax1.scatter(x1, y1, z1, c=c1, cmap=plt.hot())
    fig.colorbar(img1, ax=ax1, orientation='vertical')
    ax1.set_xlim3d(-10, 10)
    ax1.set_ylim3d(-10, 10)
    ax1.set_zlim3d(-10, 10)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('prediction')
    img2 = ax2.scatter(x2, y2, z2, c=c2, cmap=plt.hot())
    fig.colorbar(img2, ax=ax2, orientation='vertical')
    ax2.set_xlim3d(-10, 10)
    ax2.set_ylim3d(-10, 10)
    ax2.set_zlim3d(-10, 10)
    plt.show()

def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)


def plot_grid(data):
    g = sns.PairGrid(data)
    g.map_diag(sns.histplot, kde=True, stat='density', label='samples')
    # g.map_diag(summary)
    g.map_upper(sns.scatterplot, edgecolor='w')
    # g.map_lower(sns.kdeplot)
    # for ax in g.axes.flatten():
        # # rotate x axis labels
        # ax.set_xlabel(ax.get_xlabel(), rotation=0, fontsize=12)
        # ax.set_xticks(np.arange(-400,200,100))
        # ax.tick_params(axis='both', labelsize=12)
        # ax.set_ylabel(ax.get_ylabel(), rotation=90, fontsize=12)
        # ax.set_yticks(np.arange(-400,200,100))

        # loc = plticker.MultipleLocator(100)  # this locator puts ticks at regular intervals
        # ax.xaxis.set_major_locator(loc)
        # loc = plticker.MultipleLocator(100)  # this locator puts ticks at regular intervals

        # ax.yaxis.set_major_locator(loc)

        # # set y labels alignment
        # ax.yaxis.get_label().set_horizontalalignment('right')
    g.fig.set_size_inches(15, 10)

    # g.fig.suptitle('Pair plot of delta ID prices', y=1)
    plt.tight_layout()

