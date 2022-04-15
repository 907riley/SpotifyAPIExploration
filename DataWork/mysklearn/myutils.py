"""
Programmer: Riley Sikes
Class: CPSC 322-01, Spring 2021
Programming Assignment #6
3/30/22
Notes: if I dont use range(len()) my program doesn't work.
Linter tries to get me to use enumerate but it breaks my code
Description: This program contains utility functions for PA5
"""

import copy
import importlib
import math
import operator
import numpy as np
from sklearn.metrics import accuracy_score
# # uncomment once you paste your mypytable.py into mysklearn package
# import mysklearn.mypytable
# importlib.reload(mysklearn.mypytable)
# from mysklearn.mypytable import MyPyTable 

# # uncomment once you paste your myclassifiers.py into mysklearn package
# # import mysklearn.myclassifiers
# # importlib.reload(mysklearn.myclassifiers)
# # from myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier

# import mysklearn.myevaluation
# importlib.reload(mysklearn.myevaluation)
# import mysklearn.myevaluation as myevaluation


def compute_euclidean_distance(v1, v2):
    """Computes the euclidean distance between two values
        v1 (numeric value): first value
        v2 (numeric value): second value
    Returns:
        dist: The distance between the two values
    """
    dist = 0
    sum_ = 0
    if len(v1) != 0 and len(v2) != 0 and (isinstance(v1[0], int) or (isinstance(v1[0], float))):
        for i in range(len(v1)):
            sum_ += ((v1[i] - v2[i]) ** 2)

        dist = (sum_) ** (1/2)
    else:
        if v1 != v2:
            dist = 1
    return dist

def auto_discretizer(x):
    """Runs a list of mpg values through a discretize to put them into categories
    Args:
        x (list): the list of values to discretize
    Returns:
        classification: The list of discretizied values
    """
    classification = []
    for val in x:
        if val > 44:
            classification.append(10)
        elif val > 36:
            classification.append(9)
        elif val > 30:
            classification.append(8)
        elif val > 26:
            classification.append(7)
        elif val > 23:
            classification.append(6)
        elif val > 19:
            classification.append(5)
        elif val > 16:
            classification.append(4)
        elif val > 14:
            classification.append(3)
        elif val > 13:
            classification.append(2)
        else:
            classification.append(1)
    return classification


def normalize(x):
    """Normalizes a list of numeric values
    Args:
        x (list): the list of values to normalize
    Returns:
        x: The list of normalized values
    """
    max_val = max(x)
    min_val = min(x)

    for i in range(len(x)):
        x[i] = (x[i] - min_val)/(max_val - min_val)

    return x


def randomize_in_place(alist, parallel_list=None):
    """Randomizes a list in place, also randomizes a parallel list the same way
    Args:
        alist (list): the list of values to shuffle
        parallel_list (list): the parallel_list to shuffle the same way
    """
    # test functions fails if the size is two and they just swap back and forth
    if len(alist) > 2:
        for i in range(len(alist)):
            # generate a random index to swap this value at i with
            rand_index = np.random.randint(0, len(alist)) # rand in in [0, len(alist))
            # do the swap
            alist[i], alist[rand_index] = alist[rand_index], alist[i]
            if parallel_list is not None:
                parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]
    else:
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand in in [0, len(alist))
        # do the swap
        alist[0], alist[rand_index] = alist[rand_index], alist[0]
        if parallel_list is not None:
            parallel_list[0], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[0]

def title_line(number, title):
    """Helper to print out the step number and title
    Args:
        number (int): the list of values to normalize
        title (str)L the title of the section
    """
    print("===========================================")
    print("STEP " + number + ":" , title)
    print("===========================================")

def matrix_printer_helper(title, confusion_matrix):
    """Helper for pretty printing the confusion matrix
    Args:
        title (str): the title of the confusion matrix
        confusion_matrix (2D list): the confusion matrix
    """
    print(title + ":")
    print()
    print("  MPG RANKING    1    2    3    4    5    6    7    8    9    10    Total    Recognition (%)")
    print("_____________  ___  ___  ___  ___  ___  ___  ___  ___  ___  ____  _______  _________________")
    
    for i in range(len(confusion_matrix)):
        # print out init spaces
        j = len(str(i + 1))
        while j < 13:
            print(" ", end="")
            j += 1
        # print out ranking number
        print(i + 1, end="")

        # now print each of the confusin matrix values at i
        row_total = 0
        for j in range(len(confusion_matrix[i])):
            space = len(str(confusion_matrix[i][j]))
            # print out spaces
            while space < 5:
                print(" ", end="")
                space += 1
            # print out confusion matrix value at that index
            print(confusion_matrix[i][j], end="")
            row_total += confusion_matrix[i][j]
        
        # print out total acconting for space
        space = len(str(row_total))
        while space < 9:
            print(" ", end="")
            space += 1
        print(row_total, end="")

        # print out recognition %
        recog = 0.0
        if row_total != 0:
            recog = round((confusion_matrix[i][i] / row_total) * 100, ndigits=2)
        space = len(str(recog))
        while space < 19:
            print(" ", end="")
            space += 1
        print(recog)

def matrix_titanic_helper(title, confusion_matrix, labels):
    """Helper for pretty printing the confusion matrix
    Args:
        title (str): the title of the confusion matrix
        confusion_matrix (2D list): the confusion matrix
    """
    print(title + ":")
    print()
    print("    TITANIC     no   yes   Total    Recognition (%)")
    print("_____________  ___  ___  _______  _________________")
    
    for i in range(len(confusion_matrix)):
        # print out init spaces
        j = len(labels[i])
        while j < 13:
            print(" ", end="")
            j += 1
        # print out ranking number
        print(labels[i], end="")

        # now print each of the confusin matrix values at i
        row_total = 0
        for j in range(len(confusion_matrix[i])):
            space = len(str(confusion_matrix[i][j]))
            # print out spaces
            while space < 5:
                print(" ", end="")
                space += 1
            # print out confusion matrix value at that index
            print(confusion_matrix[i][j], end="")
            row_total += confusion_matrix[i][j]
        
        # print out total acconting for space
        space = len(str(row_total))
        while space < 9:
            print(" ", end="")
            space += 1
        print(row_total, end="")

        # print out recognition %
        recog = 0.0
        if row_total != 0:
            recog = round((confusion_matrix[i][i] / row_total) * 100, ndigits=2)
        space = len(str(recog))
        while space < 19:
            print(" ", end="")
            space += 1
        print(recog)

# THIS WAS CAUSING CIRCULAR IMPORTING
# def stratified_classifier_test(classifier, X, y, X_train_folds, X_test_folds):
#     """Function for PA6, since all three classifiers do the same thing, just with
#         a different classifier.
#     Args:
#         classifier (string): the name of the classifier
#         X (list of lists): the X values
#         y (list): the y values
#         X_train_folds (list of lists): the indexes of the train values to use
#         X_test_folds (list of lists): the indexes of the test values to use
#     Returns:
#         accuracy_total: the total accuracy
#         precision_total: the total precision
#         recall_total: the total recall
#         f1_total: the f1 total
#         accuracy_count: the total count of correct predictions
#         total_preds: the total predictions
#         sk_fold_y_true: all the true classes
#         sk_fold_y_pred: all the predicted classes
#     """
#     accuracy_total = 0
#     precision_total = 0
#     recall_total = 0
#     f1_total = 0
#     accuracy_count = 0

#     total_preds = 0

#     # save these for confusion matrix
#     sk_fold_y_true = []
#     sk_fold_y_pred = []

#     for i in range(len(X_train_folds)):
#         # build up the X_train, X_test, y_train, y_test
#         X_train = [X[X_train_folds[i][j]] for j in range(len(X_train_folds[i]))]
#         y_train = [y[X_train_folds[i][j]] for j in range(len(X_train_folds[i]))]

#         X_test = [X[X_test_folds[i][j]] for j in range(len(X_test_folds[i]))]
#         y_test = [y[X_test_folds[i][j]] for j in range(len(X_test_folds[i]))]

#         # do the classifying
#         if classifier == "dummy":
#             classifier = MyDummyClassifier()
#         elif classifier == "naive bayes":
#             classifier = MyNaiveBayesClassifier()
#         elif classifier == "knn":
#             classifier = MyKNeighborsClassifier()

#         classifier.fit(X_train, y_train)
#         y_pred = classifier.predict(X_test)

#         accuracy_total += myevaluation.accuracy_score(y_test, y_pred, normalize=False)
#         precision_total += myevaluation.binary_precision_score(y_test, y_pred)
#         recall_total += myevaluation.binary_recall_score(y_test, y_pred)
#         f1_total += myevaluation.binary_f1_score(y_test, y_pred)
#         accuracy_count += myevaluation.accuracy_score(y_test, y_pred, normalize=False)

#         # append preds
#         for i in range(len(y_test)):
#             sk_fold_y_true.append(y_test[i])
#             sk_fold_y_pred.append(y_pred[i])

#         total_preds += len(y_test)


#     return accuracy_total, precision_total, recall_total, f1_total, accuracy_count, total_preds, \
#            sk_fold_y_true, sk_fold_y_pred


def select_attributes(instances, attributes):
    """select the attribute to split on using entropy
    Args:
        instances (2D list): the remaining instances
        attributes (list): the attributes left
    Returns:
        entropy[0][0]: the attribute with the smallest entropy
    """
    # print("In select att", instances)
    # with the smallest E_new
    # get the class vals for computing enews
    class_vals = []
    for i, row in enumerate(instances):
        if [row[-1], 0] not in class_vals:
            class_vals.append([row[-1], 0])
    # print(class_vals)

    # general Enew algorithm pseudocode
    e_news = {}
    # for each available attribute:
    for i, att in enumerate(attributes):
        # print("start of loop",att, attributes)
        # build up the attribute domains for the current instances
        domain_values = {}
    #   for each attribute value in the domain:
        # for j, val in enumerate(domain_values):
    #       Compute the entropy of that value partition (e.g. proportion and log for each class) (function)
        for j, instance in enumerate(instances):
            # print("befoer the check", instance[int(att[3:])], domain_values)
            if instance[int(att[3:])] not in domain_values:
                # print("after the check", instance[int(att[3:])], domain_values)
                # compute the entropy
                # print("b4", class_vals)
                cp_class_vals = copy.deepcopy(class_vals)
                # print("pasing into entrop", att, instance[int(att[3:])])
                entropy = compute_entropy(instances, att, instance[int(att[3:])], cp_class_vals)
                # print(att, entropy)
                domain_values[instance[int(att[3:])]] = entropy
        e_news[att] = domain_values
    #   Compute Enew by taking weighted sum of the partition entropies
        total_e_new = 0
        # print(e_news)
        for key in e_news[att]:
            total_e_new += e_news[att][key][0] * e_news[att][key][1] / len(instances)
        # print(att, total_e_new)
        # if total_e_new == 0:
        #     del e_news[att]
        # else:
        e_news[att] = total_e_new

    # Choose to split on the attribute with the smallest Enew  
    list_entropy = list(e_news.items())
    list_entropy.sort(key=operator.itemgetter(-1, 0))
    # print("attributes + entropies", attributes, list_entropy) 
    if len(list_entropy) > 0:
        # print("-------------SPLITTING ON", list_entropy[0][0])
        return list_entropy[0][0]
        
    # error, no entropies
    return -1

    # for now, we will just choose randomly
    # rand_index = np.random.randint(0, len(attributes))
    # return attributes[rand_index]

def compute_entropy(instances, att, att_val, class_vals):
    """computes the entropy for a set of instances for all
        possible class values and attributes
    Args:
        instances (2D list): the remaining instances
        att (str): the att work with
        att_val (str): the current value to work on
        class_vals (2D list): the possible class vals
    Returns:
        [entropy, denom]: entropy and the denom for computing weighted entropy
    """
    # print("in compute entropy", instances, att_index, att_val, class_vals)
    # get the denom
    denom = 0
    for i, val in enumerate(instances):
        # print(val)
        if val[int(att[3:])] == att_val:
            # find the index of the same class
            for j in range(len(class_vals)):
                if class_vals[j][0] == val[-1]:
                    class_vals[j][1] += 1
            denom += 1
    # print("in compute entrop", att, att_val, class_vals)

    if denom == 0:
        return 0
    entropy = 0
    # now divide by denom to get the p values
    # and compute entropy
    for j in range(len(class_vals)):
        class_vals[j][1] /= denom
        p_yes = class_vals[j][1]
        # print("computing p_yes", class_vals)
        if p_yes > 0:
            entropy += p_yes * math.log(p_yes, 2)
    # make negative
    if entropy != 0:
        entropy *= -1
    # print("computed entropy for", att, entropy, denom)
    return [entropy, denom]


def partition_instances(instances, split_attribute, attribute_domains):
    """partitions instances into matching on an attribute
    Args:
        instances (2D list): the remaining instances
        split_attribute (list): the att to split on
        attribute_domains (list): the domain atts and values
    Returns:
        partitions (2D list): the partitions of the current instances
    """
    # lets use a dictionary
    partitions = {} # key (string): value (subtable)
    # print(split_attribute)
    # att_index = header.index(split_attribute) # e.g. 0 for level
    # att_index = int(split_attribute[3:]) # e.g. 0 for level

    att_domain = attribute_domains[split_attribute] # e.g. ["Junior", "Mid", "Senior"]
    att_domain.sort()
    # print(att_domain)

    for att_value in att_domain:
        partitions[att_value] = []
        for instance in instances:
            if instance[int(split_attribute[3:])] == att_value:
                partitions[att_value].append(instance)
        # task: finish

    return partitions

def all_same_class(att_partition):
    """helper for checking to see if a partion is all the same class
    Args:
        att_partition (2D list): att instances
    Returns:
        bool: True if all matching, False otherwise
    """
    # return true if all the same class label
    # class label at -1
    if len(att_partition) > 0:
        prev = att_partition[0][-1]
        for value in att_partition:
            if prev != value[-1]:
                return False
    return True

def tdidt(current_instances, available_attributes, gen_attribute_domains):
    """the main function for building up the tree
    Args:
        current_instances (2D list): the remaining instances
        available_attributes (list): the available atts left
        gen_attribute_domains (list): the domain atts and values
    Returns:
        tree: the entire or partial tree, it's recursive
    """
    # basic approach (uses recursion!!):
    # print("available attributes:", available_attributes)

    # select an attribute to split on
    attribute = select_attributes(current_instances, available_attributes)
    # print("splitting on attribute:", attribute)
    available_attributes.remove(attribute)
    tree = ["Attribute", attribute]
    # group data by attribute domains (creates pairwise disjoint partitions)
    # print("right before partitions", attribute)
    partitions = partition_instances(current_instances, attribute, gen_attribute_domains)
    # print("partitions:", partitions.items())
    # for each partition, repeat unless one of the following occurs (base case)

    for att_value, att_partition in partitions.items():
        # print("current attribute value:", att_value, len(att_partition))
        value_subtree = ["Value", att_value]
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(att_partition) > 0 and all_same_class(att_partition):
            # print("CASE 1 all same class")
            # TODO: make a leaf node
            leaf = ["Leaf", att_partition[0][-1], len(att_partition), len(current_instances)]
            value_subtree.append(leaf)
            tree.append(value_subtree)
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            # print("CASE 2 no more attributes")
            # print(current_instances)
            class_vals = []
            for i, row in enumerate(current_instances):
                if [row[-1], 0] not in class_vals:
                    class_vals.append([row[-1], 0])
            for i, row in enumerate(current_instances):
                for j in range(len(class_vals)):
                    if class_vals[j][0] == row[-1]:
                        class_vals[j][1] += 1
            class_vals.sort(key=operator.itemgetter(0))
            # print(class_vals)
            tree = ["Leaf", class_vals[0][0], len(current_instances), len(current_instances)]
            # find the majority
            # TODO: we have a mix of labels, handle clash with majority vote leaf node
        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(att_partition) == 0:
            # print("CASE 3 empty partition")
            pass
            # TODO: "backtrack" to replace the attribute node
            # with a majority vote leaf node
            # change tree to be a majority vote leaf node instead of a branch
        else: # the previous conditions are all false... recurse!
            subtree = tdidt(att_partition, available_attributes.copy(), gen_attribute_domains)
            # note the copy
            # TODO: append subtree to value_subtree and to tree
            # if the subtree has Value, then we know we have a case 3 on recurse
            # look at the len(instances) for denom here
            # print("SUBTRESS", subtree, current_instances, available_attributes)
            if subtree[0] == "Leaf":
                subtree[3] = len(current_instances)
            value_subtree.append(subtree)
            tree.append(value_subtree)
            # appropriately
    return tree

def extract_attribute_domains(instances):
    """helper function for extracting the attribute domains
    Args:
        instances (2D list): all the instances
    Returns:
        gen_attribute_domains (dict): dict to lists for atts to all poss
                                      att values
    """
    gen_attriute_domains = {}
    for i in range(len(instances[0])):
        gen_attriute_domains["att" + str(i)] = []
        for j in range(len(instances)):
            if instances[j][i] not in gen_attriute_domains["att" + str(i)]:
                gen_attriute_domains["att" + str(i)].append(instances[j][i])
    return gen_attriute_domains            


def fit_starter_code(X_train, y_train):
    """just a runner function for tdidt
    Args:
        X_train (2D list): the X_train values
        y_train (list): the y_train values
    Returns:
        tree: the entire tree
    """
    # TODO: programmatically extract the header (e.g. ["att0", "att1", ...])
    gen_attribute_domains = extract_attribute_domains(X_train)
    # print(gen_attribute_domains)
    # and atract the attribute domains
    # now, I advise stitching X_train and y_train together
    train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
    # next, make a copy of your header... tdidt() is going
    # to modify the list
    # available_attributes = header.copy()
    gen_atts = []
    for i, att in enumerate(X_train[0]): 
        new_att = "att" + str(i)
        gen_atts.append(new_att)
    # print(gen_atts)
    # also: recall that python is pass by object reference
    tree = tdidt(train, gen_atts, gen_attribute_domains)
    # print(tree)
    return tree
    # note: unit test is going to assert that tree == interview_tree_solution
    # mind (the attribute domain ordering)

def traverse_tree(tree, x):
    """traverse_tree recursively
    Args:
        tree (tree list): the tree to traverse
        x (list): the attributes to traverse down
    Returns:
        tree[1]: or the leaf node instance that it predicts
    """
    if tree[0] == "Leaf":
        # print("returning this value", tree[1])
        return tree[1]
    # print("tree: ", tree)

    index = int((tree[1])[3:])
    # print(index, x)

    # for each possible value for the attribute
    # check to see which one matches and return that tree
    for i in range(2, len(tree)):
        if tree[i][1] == x[index]:
            return traverse_tree(tree[i][2], x)


def print_decision_rules_recursive(tree, rules, attribute_names=None, class_name="class"):
    """recursive function for printing decision rules
    Args:
        tree (tree list): the tree to traverse
        rules (list): the current rule building
        attribute_names (list): the optional attribute names
        class_name (str): the optional class_name
    """

    if tree[0] == "Leaf":
        # print("completed a rule", tree[1])
        rules.append(class_name + " = " + str(tree[1]))
        for i in range(len(rules) - 1):
            if i == len(rules) - 2:
                print(rules[i] + " THEN ", end="")
            else:
                print(rules[i] + " AND ", end="")
        # print(rules[len(rules) - 1] + " THEN")
        print(rules[len(rules) - 1])
        return
    

    # index = int((tree[1])[3:])
    # print("Outer Rules:", rules)

    # for each possible value for the attribute
    # go down that path and add to the current rule
    
    for i in range(2, len(tree)):
        new_rules = rules.copy()
        # add the att to the rules and go down the path
        if attribute_names is not None:
            index = int(tree[1][3:])
            new_rules.append(attribute_names[index] + " == " + str(tree[i][1]))
        else:
            new_rules.append(str(tree[1]) + " == " + str(tree[i][1]))
        print_decision_rules_recursive(tree[i][2], new_rules, attribute_names, class_name)



def basketball_titanic_helper(title, confusion_matrix, labels):
    """Helper for pretty printing the confusion matrix
    Args:
        title (str): the title of the confusion matrix
        confusion_matrix (2D list): the confusion matrix
    """
    print(title + ":")
    print()
    print(" BASKETBALL     A   H   Total    Recognition (%)")
    print("_____________  ___  ___  _______  _________________")
    
    for i in range(len(confusion_matrix)):
        # print out init spaces
        j = len(labels[i])
        while j < 13:
            print(" ", end="")
            j += 1
        # print out ranking number
        print(labels[i], end="")

        # now print each of the confusin matrix values at i
        row_total = 0
        for j in range(len(confusion_matrix[i])):
            space = len(str(confusion_matrix[i][j]))
            # print out spaces
            while space < 5:
                print(" ", end="")
                space += 1
            # print out confusion matrix value at that index
            print(confusion_matrix[i][j], end="")
            row_total += confusion_matrix[i][j]
        
        # print out total acconting for space
        space = len(str(row_total))
        while space < 9:
            print(" ", end="")
            space += 1
        print(row_total, end="")

        # print out recognition %
        recog = 0.0
        if row_total != 0:
            recog = round((confusion_matrix[i][i] / row_total) * 100, ndigits=2)
        space = len(str(recog))
        while space < 19:
            print(" ", end="")
            space += 1
        print(recog)
