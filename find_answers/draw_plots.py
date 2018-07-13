# -*- coding: utf-8 -*-
#
# Using ~/anaconda3/bin/python: Python 3.5.2 :: Anaconda 4.2.0 (64-bit)
# Using Python 3.4.5 :: Anaconda 4.3.0 (64-bit), since Tue2017_0710
# Requires Python 3; does not work w/ Python 2.
#
"""draw_plots.py


Draw plots from the i/p data provided by the caller.



Reference:

    This file is part of the learn_python/find_answers project
    at this URL:

        https://github.com/clp/learn_python/tree/master/find_answers



Usage:

    import draw_plots as dr
    pydoc  draw_plots

    See the fga_find_good_answers.py program for an example that
    uses this module.



Initialization and Operation:

    Run the fga*.py program to build the required data set.

    When prompted from the fga menu, choose one of the actions
    that draws a plot, eg, "d: draw default plot of current data".


----------------------------------------------------------


Input data format.

    TBD

Output data format.

    TBD

----------------------------------------------------------


Requirements
    Python 3, tested with v 3.6.1.
    pytest for tests, tested with v 3.0.7.

"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
import os
import pandas as pd
from pandas.plotting import scatter_matrix

import config as cf
import util.write as wr


cf.logger.info(
    cf.log_file +
    ' - Start logging for ' +
    os.path.basename(__file__))


DATADIR = cf.DATADIR



def draw_histogram_plot(plot_df):
    """Draw a simple histogram plot using pandas tools.
    """
    fig, ax = plt.subplots(1, 1)
    ax.get_xaxis().set_visible(True)
    plot_df = plot_df[['Score']]
    # TBD These custom sized bins are used for debugging; change later.
    # histo_bins = [-10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    #               13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50]
    histo_bins = [-10, -9, -8, -7, -6, -5, -4, -3, -
                  2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    plot_df.plot.hist(ax=ax, figsize=(6, 6), bins=histo_bins)
    plt.show(block=False)

    # Write data set to a csv file.
    outfile = DATADIR + 'dh_draw_histogram.csv'
    plot_df['Score'].to_csv(
        outfile,
        header=True,
        index=None,
        sep=',',
        mode='w')

    # Print data used for histogram
    count, division = np.histogram(plot_df['Score'], bins=histo_bins)
    print("#D histogram data: count[:5]: ", count[:5])
    print("#D histogram data: division[:5]: ", division[:5])
    return


def draw_scatter_matrix_plot(plot_df):
    """Draw a set of scatter plots showing each feature vs
    every other feature.
    """
    cf.logger.info('Summary stats from plot_df.describe(): ')
    cf.logger.info(plot_df.describe())

    print('NOTE: Please wait for plot, 60 sec or more.')

    axs = scatter_matrix(plot_df, alpha=0.2, diagonal='hist')
    # TBD Failed. Logarithm scale, Good to show outliers. Cannot show Score=0?
    # plt.xscale('log')

    plt.show(block=False)

    wfile = 'scat_mat_plot.pdf'
    wr.save_prior_file(DATADIR, wfile)
    plt.savefig(DATADIR + wfile)

    wfile = 'scat_mat_plot.png'
    wr.save_prior_file(DATADIR, wfile)
    plt.savefig(DATADIR + wfile)
    return


def draw_scatter_plot(plot_df, xaxis, yaxis, xname, yname):
    """Draw a simple scatter plot using pandas tools.
    """
    ax = plot_df[[yname, xname]].plot.scatter(x=xaxis, y=yaxis, table=False)
    plt.show(block=False)
    return

