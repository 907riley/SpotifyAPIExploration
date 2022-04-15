##############################################
# Programmer: Riley Sikes
# Class: CPSC 322-01, Spring 2022
# Programming Assignment #3
# 2/22/22
# No comments
# No bonus 
#  
# Description: This file contains functions
# used for plotting in the notebooks for PA3
##############################################

import numpy as np
import matplotlib.pyplot as plt
import utils

def bar_chart_example(x, y, title, x_label, y_label):
    """Creates a bar chart from inputs
    
    Args:
        x (list): x values
        y (list): y values
        title (string): title of the graph
        x_label (string): label for the x axis
        y_label (string): label for the y axis
    
    Returns: 
        displays the graph
    """ 

    plt.figure(figsize=(15, 10))
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=30, ha="right") # parallel lines
    plt.grid(True)
    

    plt.show()

def pie_chart_example(x, y):
    """Creates a pie chart from inputs
    
    Args:
        x (list): labels
        y (list): percent values

    
    Returns: 
        displays the graph
    """ 

    plt.figure()
    plt.pie(y, labels=x, autopct="%1.1f%%", normalize=False)
    plt.show()

def histogram_helper(data, title):
    """Creates a histogram from inputs
    
    Args:
        data (list): the data to be broken up into 
            10 bins
        title (string): the title of the graph
    
    Returns: 
        displays the graph
    """ 

    plt.figure()
    plt.hist(data, bins=10, edgecolor="black")
    plt.title(title)
    plt.grid(True)
    plt.show()

def scatter_helper(x, y, title, xlabel, ylabel):
    """Creates a scatter plot with a linear regression
        line over the top of the points. Also displays
        the correlation coefficient and covaraince for the
        linear regression model in the top right corner
        of the graph.
    
    Args:
        x (list): x values
        y (list): y values
        title (string): title of the graph
        x_label (string): label for the x axis
        y_label (string): label for the y axis
    
    Returns: 
        displays the graph
    """ 

    plt.figure()
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    m, b = utils.compute_slope_intercept(x, y)
    plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c="r", lw=3)
    plt.grid(True)

    cor = utils.compute_correlation_coefficient(x, y)
    cov = utils.compute_corvariance(x, y)

    ax = plt.gca()
    ax.annotate("cor=%.2f, cov=%.2f" %(cor, cov),
        xy=(.61, .93), xycoords='axes fraction', color="r",
        bbox=dict(boxstyle="round", fc="1", color="r"))    
    plt.show()

def scatter_helper_lim(x, y, title, xlabel, ylabel, xlim, ylim):
    """Creates a scatter plot with a linear regression
        line over the top of the points. Also displays
        the correlation coefficient and covaraince for the
        linear regression model in the top right corner
        of the graph. 

        Also takes in xlim and ylim for manually setting
        axis ranges
    
    Args:
        x (list): x values
        y (list): y values
        title (string): title of the graph
        x_label (string): label for the x axis
        y_label (string): label for the y axis
        xlim (int): upper bound for x axis
        ylim (int): upper bound for y axis
    
    Returns: 
        displays the graph
    """  

    plt.figure()
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0, xlim)
    plt.ylim(0, ylim)
    m, b = utils.compute_slope_intercept(x, y)
    plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c="r", lw=3)
    plt.grid(True)

    cor = utils.compute_correlation_coefficient(x, y)
    cov = utils.compute_corvariance(x, y)

    ax = plt.gca()
    ax.annotate("cor=%.2f, cov=%.2f" %(cor, cov),
        xy=(.61, .93), xycoords='axes fraction', color="r",
        bbox=dict(boxstyle="round", fc="1", color="r"))    
    plt.show()

def box_and_whiskers_helper(data, xlabels, title):
    """Creates a box and whisker plot from inputs
    
    Args:
        data (list): list of lists for creating multiple
            box and whiskers graphs on the same figure
        xlabels (list): list of labels for each graph
        title (string): title of the graph
    
    Returns: 
        displays the graph
    """  

    plt.figure()
    plt.figure(figsize=(15, 10))
    plt.title(title)
    plt.boxplot(data)
    plt.xticks([i for i in range(1, len(xlabels) + 1)], xlabels)
    plt.xticks(rotation=30, ha="right") # parallel lines
    plt.show()

def box_and_whiskers_helper_ylim(data, xlabels, title, ylim):
    """Creates a box and whisker plot from inputs

        Also takes in ylim for manually setting
        axis range
    
    Args:
        data (list): list of lists for creating multiple
            box and whiskers graphs on the same figure
        xlabels (list): list of labels for each graph
        title (string): title of the graph
        ylim (int): upper bound for y axis
    
    Returns: 
        displays the graph
    """  

    plt.figure()
    plt.figure(figsize=(15, 10))
    plt.ylim(0, ylim)
    plt.title(title)
    plt.boxplot(data)
    plt.xticks([i for i in range(1, len(xlabels) + 1)], xlabels)
    plt.xticks(rotation=30, ha="right") # parallel lines
    plt.show()
