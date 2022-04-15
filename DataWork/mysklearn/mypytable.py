##############################################
# Programmer: Riley Sikes
# Class: CPSC 322-01, Spring 2021
# Programming Assignment #2
# 2/9/20
# No notes
# 
# Description: This program defines the MyPyTable
# class that implements a series of helper functions 
# for handling data sets
##############################################
from mysklearn import myutils
from ast import And
import copy
import csv
from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """

        try:
            col_index = self.column_names.index(col_identifier)
        except ValueError:
            raise ValueError("ValueError thrown")
        col = []
        for row in self.data:
            value = row[col_index]
            if include_missing_values == True or value != "NA":
                col.append(value)

        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data:
            for i in range(len(row)):
                try: 
                    numeric_value = float(row[i])

                    row[i] = numeric_value
                except ValueError:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        row_indexes_to_drop.sort(reverse=True)
        for value in row_indexes_to_drop:
            self.data.pop(value)
    
    def drop_cols(self, col_indexes_to_drop):
        """Remove columns from the table data.

        Args:
            col_indexes_to_drop(list of int): list of column indexes to remove from the table data.
        """
        col_indexes_to_drop.sort(reverse=True)
        
        for val in col_indexes_to_drop:
            del self.column_names[val]

        for i in range(len(self.data)):
            for val in col_indexes_to_drop:
                del self.data[i][val]

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        infile = open(filename, "r")

        csvreader = csv.reader(infile)

        self.column_names = next(csvreader)

        for row in csvreader:
            self.data.append(row)

        self.convert_to_numeric()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """

        outfile = open(filename, "w")

        for i in range(len(self.column_names) - 1):
            outfile.write(str(self.column_names[i]) + ",")
        outfile.write(str(self.column_names[-1]) + "\n")

        for row in self.data:
            for j in range(len(row) - 1):
                outfile.write(str(row[j]) + ",")
            outfile.write(str(row[-1]) + "\n")
        outfile.close()

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """

        # convert the key_column_names to a list of indexes
        key_cols = []
        key_pairs = []
        dups = []

        for val in key_column_names:
            key_cols.append(self.get_column(val))

        # just need to go through the length of a column
        for i in range(len(self.data)):
            key_pairs.append([])

        # now go through each of the key_columns and create composite keys
        for i in range(len(key_cols)):
            for j in range(len(key_cols[i])):
                key_pairs[j].append(key_cols[i][j])

        # now go through and build up a scene list, if not in list add
        # if in list add to dupes and move on
        seen_keys = []
        for i in range(len(key_pairs)):
            if seen_keys.count(key_pairs[i]) == 0:
                seen_keys.append(key_pairs[i])
            else:
                dups.append(i)

        return dups

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """

        for i in reversed(range(len(self.data))):
            for j in range(len(self.data[i])):
                if self.data[i][j] == "NA":
                    self.data.remove(self.data[i])


    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        # get the column
        col = self.get_column(col_name)

        col_index = self.column_names.index(col_name)

        # make sure the data is continous
        if len(col) > 0:
            if type(col[0]) == int or type(col[0]) == float:
                # now go through the col and get the avg when the val isn't "NA"
                total = 0
                count = 0
                for i in range(len(col)):
                    if col[i] != "NA":
                        total += col[i]
                        count += 1
                avg = total / count
                # now that we have the avg, go through and replace "NA"
                for i in range(len(col)):
                    if col[i] == "NA":
                        self.data[i][col_index] = avg
                

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        stats_header = ["attribute", "min", "max", "mid", "avg", "median"]
        full_stats = []

        for val in col_names:
            col = self.get_column(val)
            stats = []
            # drop the values that have an 'NA'
            # so we can do stats
            for i in reversed(range(len(col))):
                if col[i] == 'NA':
                    col.pop(i)
            # now check to make sure continuous
            if len(col) > 0 and (type(col[0]) == float or type(col[0]) == int):
                # now go through and compute the stats
                stats.append(val)
                stats.append(min(col))
                stats.append(max(col))
                stats.append((max(col) + min(col)) / 2)
                total = 0
                count = 0
                for value in col:
                    if value != "NA":
                        total += value
                        count += 1
                stats.append(total/count)
                col.sort()
                if len(col) % 2 == 0:
                    stats.append((col[int((len(col) / 2) + .5)] + col[int((len(col) / 2) - .5)]) / 2)
                else:
                    stats.append(col[int(len(col) / 2)])
                full_stats.append(stats)

        new_table = MyPyTable(stats_header, full_stats)
        return new_table

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        # go through each row in the first table and see if the composite
        # key matches any keys in the other table
        # if there is a match, add the composite key to the new table with the other attributes
        # in the same row
        # convert the key_column_names to a list of indexes
        key_cols = []
        key_pairs = []

        for val in key_column_names:
            key_cols.append(self.get_column(val))

        # just need to go through the length of a column
        for i in range(len(self.data)):
            key_pairs.append([])

        # now go through each of the key_columns and create composite keys
        for i in range(len(key_cols)):
            for j in range(len(key_cols[i])):
                key_pairs[j].append(key_cols[i][j])
        
        other_key_cols = []
        other_key_pairs = []

        for val in key_column_names:
            other_key_cols.append(other_table.get_column(val))

        # just need to go through the length of a column
        for i in range(len(other_table.data)):
            other_key_pairs.append([])

        # now go through each of the key_columns and create composite keys
        for i in range(len(other_key_cols)):
            for j in range(len(other_key_cols[i])):
                other_key_pairs[j].append(other_key_cols[i][j])
        
        # copy over the previous headers from left
        new_headers = copy.deepcopy(self.column_names)
        new_data = []

        # now get the ones that are on the right that aren't on the left
        for val in other_table.column_names:
            if val not in new_headers:
                new_headers.append(val)

        # go through each instance in the left keys and see if they match the right
        # key
        for i in range(len(key_pairs)):
            for j in range(len(other_key_pairs)):
                if key_pairs[i] == other_key_pairs[j]:
                    # we have a match
                    # so now build up the new data
                    # making sure that we don't grab duplicate data
                    new_row = []
                    for q in range(len(self.column_names)):
                        new_row.append(self.data[i][q])
                    for q in range(len(other_table.column_names)):
                        if other_table.column_names[q] not in self.column_names:
                            new_row.append(other_table.data[j][q])
                    new_data.append(new_row)
        
        new_mypytable = MyPyTable(new_headers, new_data)

        return new_mypytable

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        key_cols = []
        key_pairs = []

        for val in key_column_names:
            key_cols.append(self.get_column(val))

        # just need to go through the length of a column
        for i in range(len(self.data)):
            key_pairs.append([])

        # now go through each of the key_columns and create composite keys
        for i in range(len(key_cols)):
            for j in range(len(key_cols[i])):
                key_pairs[j].append(key_cols[i][j])
        
        other_key_cols = []
        other_key_pairs = []

        for val in key_column_names:
            other_key_cols.append(other_table.get_column(val))

        # just need to go through the length of a column
        for i in range(len(other_table.data)):
            other_key_pairs.append([])

        # now go through each of the key_columns and create composite keys
        for i in range(len(other_key_cols)):
            for j in range(len(other_key_cols[i])):
                other_key_pairs[j].append(other_key_cols[i][j])
        
        # copy over the previous headers from left
        new_headers = copy.deepcopy(self.column_names)
        new_data = []

        # now get the ones that are on the right that aren't on the left
        for val in other_table.column_names:
            if val not in new_headers:
                new_headers.append(val)

        # go through each instance in the left keys and see if they match the right
        # key
        for i in range(len(key_pairs)):
            flag = False
            for j in range(len(other_key_pairs)):
                if key_pairs[i] == other_key_pairs[j]:
                    # we have a match
                    # so now build up the new data
                    # making sure that we don't grab duplicate data
                    flag = True
                    new_row = []
                    for q in range(len(self.column_names)):
                        new_row.append(self.data[i][q])
                    for q in range(len(other_table.column_names)):
                        if other_table.column_names[q] not in self.column_names:
                            new_row.append(other_table.data[j][q])
                    new_data.append(new_row)
            if not flag:
                new_row = []
                for q in range(len(self.column_names)):
                    new_row.append(self.data[i][q])
                for q in range(len(other_table.column_names)):
                    if other_table.column_names[q] not in self.column_names:
                        new_row.append("NA")
                new_data.append(new_row)    

        # need to grab the other ones from the right side
        for i in range(len(other_key_pairs)):
            flag = False
            for j in range(len(key_pairs)):
                if key_pairs[j] == other_key_pairs[i]:
                    # we have a match
                    # so now build up the new data
                    # making sure that we don't grab duplicate data
                    flag = True
            if not flag:
                new_row = []
                for q in range(len(new_headers)):
                    if new_headers[q] in other_table.column_names:
                        new_row.append(other_table.data[i][other_table.column_names.index(new_headers[q])])
                    else:
                        new_row.append("NA")
                new_data.append(new_row)           
        
        new_mypytable = MyPyTable(new_headers, new_data)
        return new_mypytable


