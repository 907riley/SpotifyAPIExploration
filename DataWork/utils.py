##############################################
# Programmer: Riley Sikes
# Class: CPSC 322-01, Spring 2022
# Programming Assignment #3
# 2/22/22
# No comments
# No bonus 
#  
# Description: This file contains general functions
# used in the notebooks for PA3
##############################################


from cmath import sqrt
import numpy as np

def get_parallel_arrays(x):
    """Builds up parallel arrays: one with the unique instances in x
        and one with the counts of each unique instance.
    
    Args:
        x (list): list of values to go through
    
    Returns: 
        bar_names: list of instances
        counts: list of counts
    """

    bar_names = []
    counts = []

    for i in range(len(x)):
        x[i] = str(x[i])


    for val in x:
        if val not in bar_names:
            bar_names.append(val)
            counts.append(1)
        else:
            index = bar_names.index(val)
            counts[index] = counts[index] + 1

    return bar_names, counts

def compute_bin_frequencies(values, cutoffs):
    """Computes the bin frequencies for a list of values

    Source:
        Taken from our in class notes on histograms
    
    Args:
        values (list): list of values compute on
        cutoffs (list): list of cutoffs to use
    
    Returns: 
        freqs: list of frequencies
    """
    freqs = [0 for _ in range(len(cutoffs) - 1)] # because we have N + 1 cutoffs

    for value in values:
        if value == max(values):
            freqs[-1] += 1 # add one to the last bin count
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= value < cutoffs[i + 1]:
                    freqs[i] += 1 
                    # add one to this bin defined by [cutoffs[i], cutoffs[i+1]]
    return freqs

def compute_equal_width_cutoffs(values, num_bins):
    """Computes the equal width cutoffs for a list of values and 
        number of bins

    Source:
        Taken from our in class notes on histograms
    
    Args:
        values (list): list of values compute on
        num_bins (int): number of bins to use
    
    Returns: 
        cutoff: list of cutoffs for the bins
    """

    values_range = max(values) - min(values)
    bin_width = values_range / num_bins # float
    # since bin_width is a float, we shouldn't use range() to generate a list
    # of cutoffs, use np.arange()
    cutoffs = list(np.arange(min(values), max(values), bin_width))
    cutoffs.append(max(values)) # exactly the max(values)
    # to handle round off error...
    # if your application allows, we should convert to int
    # or optionally round them
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs

def compute_slope_intercept(x, y):
    """Computes the slope and intercept for two lists of values

    Source:
        Taken from our in class notes on linear regression
    
    Args:
        x (list): list of values to compute on
        y (list): list of values to compute on
    
    Returns: 
        m: the slope of the line
        b: the intercept of the line
    """

    meanx = np.mean(x)
    meany = np.mean(y)

    num = sum([(x[i] - meanx) * (y[i] - meany) for i in range(len(x))])
    den = sum([(x[i] - meanx) ** 2 for i in range(len(x))])

    m = num / den
    # y = mx + b => b = y - mx
    b = meany - m * meanx
    return m, b

def compute_correlation_coefficient(x, y):
    """Computes the correlation coefficients for two lists
    
    Args:
        x (list): list of values to compute on
        y (list): list of values to compute on
    
    Returns: 
        round(correlation_coefficient, 2): the correlation
            coefficient rounded to two decimal places
    """    

    meanx = np.mean(x)
    meany = np.mean(y)

    num = sum([(x[i] - meanx) * (y[i] - meany) for i in range(len(x))])
    den = (sum([(x[i] - meanx) ** 2 for i in range(len(x))]) * sum([(y[i] - meany) ** 2 for i in range(len(y))])) ** (1/2)

    correlation_coefficient = num / den
    return round(correlation_coefficient, 2)

def compute_corvariance(x, y):
    """Computes the covariance for two lists
    
    Args:
        x (list): list of values to compute on
        y (list): list of values to compute on
    
    Returns: 
        round(covariance, 2): the covariance
         rounded to two decimal places
    """   

    meanx = np.mean(x)
    meany = np.mean(y)

    num = sum([(x[i] - meanx) * (y[i] - meany) for i in range(len(x))])
    den = len(x)

    covariance = num / den
    return round(covariance, 2)

def count_instances(data):
    """Counts the instances of the value 1 in a list
    
    Args:
        data (list): list of values to computes
    
    Returns: 
        count: the count of values that match 1
    """   

    count = 0
    for val in data:
        if val == 1:
            count += 1
    
    return count

def get_genre_ratings(genres, imdb, rotten_tomatoes):
    """Builds up three parallel lists that contain:
        - all the unique genres in genres
        - lists of ratings for each genre in genres
            by Rotten Tomatoes
        - lists of ratings for each genre in genres
            by IMDb
    
    Args:
        genres (list): the column of genres
        imdb (list): the column of imdb ratings
        rotten_tomatoes (list): the column of rotten tomatoes ratings
    
    Returns: 
        unique_genres: list of the unique genres
        rotten_tomatoes_genre_ratings: list of lists containing the rotten
            tomatoes ratings for each genre
        imdb_genre_ratings: list of lists containing the imdb
            ratings for each genre
    """   

    unique_genres = []
    rotten_tomatoes_genre_ratings = []
    imdb_genre_ratings = []

    # first go through and get all the unique genres
    for i in range(len(genres)):
        curr_genres = genres[i].split(",")
        for j in range(len(curr_genres)):
            if curr_genres[j] not in unique_genres:
                unique_genres.append(curr_genres[j])

    # init the lists
    for _ in unique_genres:
        rotten_tomatoes_genre_ratings.append([])
        imdb_genre_ratings.append([])

    # now add the score for the parallel lists
    for i in range(len(genres)):
        curr_genres = genres[i].split(",")
        for j in range(len(curr_genres)):
            # get index of genre
            index = unique_genres.index(curr_genres[j])
            # now append the score for that genre
            rotten_tomatoes_genre_ratings[index].append(rotten_tomatoes[i])
            imdb_genre_ratings[index].append(imdb[i])

    return unique_genres, rotten_tomatoes_genre_ratings, imdb_genre_ratings
