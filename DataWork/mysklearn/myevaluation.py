"""
Programmer: Riley Sikes
Class: CPSC 322-01, Spring 2021
Programming Assignment #6
3/30/22
Notes: if I dont use range(len()) my program doesn't work.
Linter tries to get me to use enumerate but it breaks my code
Description: This file has a series of functions that 
split data in different ways
"""


from mysklearn import myutils
import numpy as np

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    if random_state is not None:
        np.random.seed(random_state)
    
    if shuffle:
        myutils.randomize_in_place(X, y)

    size = 0
    # take a percentage from X and y
    if test_size < 1:
        ratio = (1 - test_size)
        size = int(len(X) * ratio)
    else:
        # take a number from X and y
        size = len(X) - test_size
        if test_size > len(X):
            size = 0

    # now do the splitting
    # grab first size for train
    for i in range(len(X)):
        if i < size:
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])

    # X_train, X_test, y_train, y_test
    return X_train, X_test, y_train, y_test

def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    if random_state is not None:
        # random_state was causing the shuffle to swap in place
        np.random.seed(random_state + 5)

    X_train_folds = []
    X_test_folds = []

    n_samples = len(X)
    curr_split = 0
    size = 0

    for i in range(n_splits):
        if i < (n_samples % n_splits):
            # size n_samples // n_splits + 1
            size = (n_samples // n_splits + 1)
        else:
            # size n_samples // n_splits
            size = (n_samples // n_splits)

        new_train_fold = []
        new_test_fold = []
        for j in range(len(X)):
            if curr_split <= j < (curr_split + size):
                new_test_fold.append(j)
            else:
                new_train_fold.append(j)
        curr_split += size
        X_train_folds.append(new_train_fold)
        X_test_folds.append(new_test_fold)

    if shuffle:
        myutils.randomize_in_place(X_train_folds, X_test_folds)

    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    if random_state is not None:
        # random_state was causing the shuffle to swap in place
        np.random.seed(random_state + 5)

    # count up the instances of each class in y
    classes = []
    for val in y:
        if [val] not in classes:
            classes.append([val])

    for i in range(len(classes)):
        classes[i].append(0)

    for val in y:
        for i in range(len(classes)):
            if classes[i][0] == val:
                classes[i][1] += 1

    # make subgroups based on what each X index has for y
    subgroups = [[] for _ in classes]

    for i in range(len(X)):
        for j in range(len(classes)):
            if y[i] == classes[j][0]:
                subgroups[j].append(i)
    
    # k fold stuff
    X_train_folds = []
    X_test_folds = []

    splits = [[] for _ in range(n_splits)]

    for i in range(len(splits)):
        curr_class = 0
        if (len(subgroups) > i):
            while len(subgroups[i]) > 0:
                splits[curr_class % n_splits].append(subgroups[i].pop(0))
                curr_class += 1

    # now add to the folds
    # need test and train folds for each split
    for i in range(n_splits):
        # now go through all the splits and grab one for test and the rest go to train
        new_train = []
        for j in range(len(splits)):
            if i == j:
                X_test_folds.append(splits[j])
            else:
                for x in range(len(splits[j])):
                    new_train.append(splits[j][x])
        X_train_folds.append(new_train)


    if shuffle:
        myutils.randomize_in_place(X_train_folds, X_test_folds)

    return X_train_folds, X_test_folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """
    samples = len(X)
    X_sample = []
    X_out_of_bag = []
    y_sample = None
    y_out_of_bag = None

    if y is not None:
        y_sample = []
        y_out_of_bag = []

    if n_samples is not None:
        samples = n_samples

    if random_state is not None:
        # random_state was causing the shuffle to swap in place
        np.random.seed(random_state)

    # add samples to the sample lists
    for _ in range(samples):
        rand_index = np.random.randint(0, len(X)) # rand int in [0, len(X))
        X_sample.append(X[rand_index])
        if y is not None:
            y_sample.append(y[rand_index])
    
    # now find the ones that weren't added
    for i in range(len(X)):
        if X[i] not in X_sample:
            X_out_of_bag.append(X[i])
            if y is not None:
                y_out_of_bag.append(y[i])

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix_solution = [[0 for _ in range(len(labels))] for _ in range(len(labels))]

    for i in range(len(y_true)):
        r = labels.index(y_true[i])
        c = labels.index(y_pred[i])
        matrix_solution[r][c] += 1

    return matrix_solution

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """

    # just go through and see how many match y_true and divide by length
    total_true = 0
    total = len(y_true)

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            total_true += 1
    
    score = total_true / total
    if not normalize:
        score = total_true
        
    return score

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if pos_label is None:
        if labels is None:
            pos_label = y_true[0]
        else:
            pos_label = labels[0]

    true_pos = 0
    false_pos = 0

    # for len y_pred
    for i in range(len(y_pred)):
    #   if y_pred == pos_label
        if y_pred[i] == pos_label:
    #       if y_pred == y_true
            if y_pred[i] == y_true[i]:
    #           tp
                true_pos += 1
    #       else
            else:
    #           fp
                false_pos += 1

    den = true_pos + false_pos
    # check for div by 0
    if den == 0:
        return 0

    return true_pos / den

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if pos_label is None:
        if labels is None:
            pos_label = y_true[0]
        else:
            pos_label = labels[0]

    true_pos = 0
    false_neg = 0

    # for len y_pred
    for i in range(len(y_true)):
    #   if y_true == pos_label
        if y_true[i] == pos_label:
    #       if y_pred == y_true
            if y_pred[i] == y_true[i]:
    #           tp
                true_pos += 1
    #       else
            else:
    #           fn
                false_neg += 1

    den = true_pos + false_neg
    # check for div by 0
    if den == 0:
        return 0

    return true_pos / den

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)

    den = precision + recall
    # div by 0 check
    if den == 0:
        return 0

    f1_score = 2 * (precision * recall) / den

    return f1_score

