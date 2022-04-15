import mysklearn.mypytable
# importlib.reload(mysklearn.mypytable)
from mysklearn.mypytable import MyPyTable

import mysklearn.myclassifiers
from mysklearn.myclassifiers import MySimpleLinearRegressor

import plot_utils
import utils

import matplotlib.pyplot as plt

def show_relationship(a, b): 

    linReg = MySimpleLinearRegressor()

    x = likedSongsTable.get_column(a)
    X_train = [[x[i]] for i in range(len(x))]

    y = likedSongsTable.get_column(b)

    linReg.fit(X_train, y)

    slope = linReg.slope
    intercept = linReg.intercept

    print("slope:", slope, "intercept:", intercept)



    # plotting the points
    plt.scatter(x, y)

    # plotting the line
    plt.plot([min(x), min(y)], [slope * min(x) + intercept, slope * max(x) + intercept], c="r", lw=2)
    
    # naming the x axis
    plt.xlabel(a)
    # naming the y axis
    plt.ylabel(b)
    
    # giving a title to my graph
    plt.title("Relationship Between " + a + " and " + b)
    
    # function to show the plot
    plt.show()

likedSongsTable = MyPyTable()

likedSongsTable.load_from_file("input_data\likedSongs.csv")

data = likedSongsTable.data

# print(likedSongsTable.column_names)
# for i in range(len(data)):
#     print(data[i])
# print(likedSongsTable.get_column("danceability"))

stats = likedSongsTable.compute_summary_statistics(["danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo"])

stats.pretty_print()


x = likedSongsTable.get_column("acousticness")
y = likedSongsTable.get_column("energy")

plot_utils.scatter_helper(x, y, "Compare acousticness and energy", "acousticness", "energy")





