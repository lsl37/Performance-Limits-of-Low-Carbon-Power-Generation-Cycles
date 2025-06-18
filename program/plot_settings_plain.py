#!/usr/bin/env python3
"""
Standard matplotlib settings and utility functions

Author: lsl37
Created: Nov 10, 2022
"""

import os
import matplotlib.pyplot as plt
from matplotlib import rc
from cycler import cycler
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


# Define color scheme
COLORLIST = ['lightgreen', '#bdee8c', '#efef88', '#efba84', 'lightcoral']

# Configure matplotlib settings
plt.rcParams['font.size'] = 16.0
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['axes.prop_cycle'] = cycler(color=COLORLIST)



def plot_polygon(ax, corners, label=None, **kwargs):
    """
    Create and plot a polygon from corner coordinates.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to draw the polygon into.
    corners : array-like
        List of (x, y) coordinate tuples defining polygon corners.
    label : str, optional
        Label for the polygon.
    **kwargs
        Additional arguments passed to PatchCollection.
    """
    poly = Polygon(corners)
    patch_collection = PatchCollection([poly], **kwargs)
    ax.add_collection(patch_collection)


