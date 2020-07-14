# Credits to
# https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
# Also to
# CS 168 helper script for Assignment 2
# http://web.stanford.edu/class/cs168/p2.pdf

from colorsys import hls_to_rgb
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import colors as mcolors
import numpy as np
from sklearn import manifold, datasets
import torch


from constants import *

KNOWN_COLORS = ['blueviolet', 'limegreen', 'darksalmon', 'cadetblue',
'lightgoldenrodyellow', 'chartreuse', 'papayawhip', 'steelblue', 'purple',
'green', 'lawngreen', 'blue', 'darkslategray', 'dodgerblue', 'indigo',
'saddlebrown', 'aquamarine', 'violet', 'lime', 'midnightblue', 'fuchsia',
'snow', 'burlywood', 'mistyrose', 'beige', 'orangered', 'darkviolet',
'mediumorchid', 'antiquewhite', 'lightblue', 'darkgrey', 'lightslategray',
'indianred', 'crimson', 'olive', 'lightsalmon', 'sandybrown', 'chocolate',
'goldenrod', 'white', 'gold', 'tan', 'plum', 'darkgreen', 'darkorange',
'palegoldenrod', 'powderblue', 'greenyellow', 'm', 'mediumvioletred', 'ivory',
'turquoise', 'cornflowerblue', 'aliceblue', 'seagreen']

def get_distinct_colors(n):
    # Fix seed every time to keep color the same for all drivers
    np.random.seed(10)

    colors = []
    for i in np.arange(0., 360., 360. / n):
        h = i / 360.
        l = (50 + np.random.rand() * 10) / 100.
        s = (90 + np.random.rand() * 10) / 100.
        colors.append(hls_to_rgb(h, l, s))

    return np.array(colors)


def get_named_colors():
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    select = ['b', 'cyan', 'darkred', 'plum', 'red',
              'green', 'mediumpurple', 'chocolate', 'yellow', 'lightslategrey',
              'lightpink', 'peachpuff', 'brown', 'lime', 'darkviolet']
    result = np.array([colors[b] for b in select])

    return result


#PLOT_COLORS = get_distinct_colors(100)
#PLOT_COLORS = get_named_colors()
PLOT_COLORS = KNOWN_COLORS


def tb_plot_prep():
    fig = Figure(dpi=128)
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    return canvas, fig, ax


def tb_plot_wrap_up(canvas, fig, ax):

    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)
    canvas.draw()       # draw the canvas, cache the renderer
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height,
                                                                        width,
                                                                        3)
    return image, fig


def plot_confusion_matrix(cm, color='YlGn'):
    canvas, fig, ax = tb_plot_prep()

    heatmap = ax.pcolor(cm, cmap=color)
    fig.colorbar(heatmap)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(cm.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(cm.shape[1]) + 0.5, minor=False)

    ax.set_xlabel('Prediction')
    ax.set_ylabel('Groud Truth')

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_title('Confusion Matrix')
    ax.set_xticklabels(range(0, NUM_DRIVERS))
    ax.set_yticklabels(range(0, NUM_DRIVERS))

    return tb_plot_wrap_up(canvas, fig, ax)


def plot_tsne(embeddings, labels, perplexity=7, n_components=2):

    canvas, fig, ax = tb_plot_prep()

    tsne = manifold.TSNE(n_components=n_components, init='pca',
                         random_state=0, perplexity=perplexity)
    Y = tsne.fit_transform(embeddings)
    # colors = PLOT_COLORS[labels]

    for d in range(NUM_DRIVERS):
        indices = [labels == d]
        y = Y[tuple(indices)]
        ax.scatter(y[:, 0], y[:, 1], s=2,
                   c=[PLOT_COLORS[d]], label=f'Driver {d}')

    ax.legend(bbox_to_anchor=(1, 1), prop={'size': 5}, loc="upper left")
    plt.tight_layout(pad=7)
    # ax.scatter(Y[:,0], Y[:,1], s=2, c=colors)
    ax.set_title(f't-SNE With Perplexity $={perplexity}$')

    return tb_plot_wrap_up(canvas, fig, ax)

def plot_tsne_collisions(embeddings, labels, perplexity=7, n_components=2):

    canvas, fig, ax = tb_plot_prep()

    tsne = manifold.TSNE(n_components=n_components, init='pca',
                         random_state=0, perplexity=perplexity)
    Y = tsne.fit_transform(embeddings)
    # colors = PLOT_COLORS[labels]
    text_legend = ['no collision','collision']
    for d in range(2):
        collision=text_legend[d]
        indices = [labels == d]
        y = Y[tuple(indices)]
        ax.scatter(y[:, 0], y[:, 1], s=2,
                   c=[PLOT_COLORS[d]], label=f'Collisions={collision}')

    ax.legend(bbox_to_anchor=(1, 1), prop={'size': 5}, loc="upper left")
    plt.tight_layout(pad=7)
    # ax.scatter(Y[:,0], Y[:,1], s=2, c=colors)
    ax.set_title(f't-SNE With Perplexity $={perplexity}$')

    return tb_plot_wrap_up(canvas, fig, ax)



