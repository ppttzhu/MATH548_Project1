# _________________LIBRARIES_________________
import numpy as np
# https://pypi.org/project/matplotlib/#files
# pip3.7 install matplotlib-2.2.3-cp37-cp37m-win32.whl
import matplotlib.pyplot as plt  # for plotting graphs
from numpy.linalg import inv  # for matrix algebra
import math


# ________________FUNCTIONS_Test______________________

def dist(x1, y1, x2, y2):
    """
    dist(float, float, float, float) --> float

    dist returns distance between
    points (x1, y1) and (x2, y2)
    """

    # definition of distance as Euclidean
    distance = (x1 - x2) ** 2 + (y1 - y2) ** 2

    return math.sqrt(distance)


def dist1(x1, y1, x2, y2):
    """
    dist(float, float, float, float) --> float

    dist returns distance between
    points (x1, y1) and (x2, y2)
    """

    # definition of distance as the addition of absolute value
    distance = abs(x1 - x2) + abs(y1 - y2)

    return distance


def dist2(x1, y1, x2, y2):
    """
    dist(float, float, float, float) --> float

    dist returns distance between
    points (x1, y1) and (x2, y2)
    """

    # definition of distance as the maximum of absolute value
    distance = max(abs(x1 - x2), abs(y1 - y2))

    return distance


def dist3(x1, y1, x2, y2):
    """
    dist(float, float, float, float) --> float

    dist returns distance between
    points (x1, y1) and (x2, y2)
    """

    # definition of distance as oral circle
    distance = 2 * ((x1 - x2) ** 2) + 3 * ((y1 - y2) ** 2)

    return math.sqrt(distance)


def dist4(x1, y1, x2, y2):  # Just to test whether rectangle will fit the model better, actually the square is better.
    """
    dist(float, float, float, float) --> float

    dist returns distance between
    points (x1, y1) and (x2, y2)
    """

    # definition of distance as the maximum of absolute value with weight
    distance = max(2 * abs(x1 - x2), 3 * abs(y1 - y2))

    return math.sqrt(distance)


def mean_cat(dists):
    """
    mean_cat(array) --> float

    mean_cat produces the averge category
    """

    sum_ = 0

    for index in range(len(dists)):
        sum_ = sum_ + dists[index][1]

    return round(sum_ / len(dists))


def optimal_cat(dists):
    """
    optimal_cat(array) --> float

    optimal_cat produces the optimal category in three category
    """

    cat1 = 0
    cat2 = 0
    cat3 = 0

    for index in range(len(dists)):
        if dists[index][1] == 1:
            cat1 += 1
        elif dists[index][1] == 2:
            cat2 += 1
        elif dists[index][1] == 3:
            cat3 += 1

    result = 1
    if cat3 > cat1 and cat3 > cat2:
        result = 3
    elif cat2 > cat1:
        result = 2

    return result


# ________________LIN REG_____________________

# ------------Data Retrieval & Prep------------

# We start by opening data file, but file is
# read as a string. So we need to turn it into an
# array of floats.


n = 0  # counter
with open("p1_data.txt", "r") as data:
    structured_data = []
    for line in data:
        # strips line of \n & splits by space
        s = line.split(" ")
        k = []
        # loop constructs strings of 3 floats per line
        for i in range(3):
            k.append(float(s[i]))
        structured_data.append(k)
        n = n + 1  # counting our lines = dimension

# Now, structured_data has all our points
# Next, we do a linear regression

# ------------Inputs & Outputs------------
# Our inputs & Outputs
m = []
y = []

for i in range(n):
    m.append([1, structured_data[i][0], structured_data[i][1]])
    y.append([structured_data[i][2]])
    # for three connected districts condition
    # if structured_data[i][2] == 1:
    #     y.append([1, 0, 0])
    # elif structured_data[i][2] == 2:
    #     y.append([0, 1, 0])
    # elif structured_data[i][2] == 3:
    #     y.append([0, 0, 1])

# ------------Matrix Algebra------------
# We want to find:
# Beta = (M^T M)^{-1} M^T Y

M = np.asmatrix(m)  # converts m into a matrix M
Y = np.asmatrix(y)  # converts y into a vector Y

prod_inv = M.transpose() * M
inverse = inv(prod_inv)

prod = M.transpose() * Y

beta = inverse * prod

# ------------Print Solution------------
# our coefficients for linear regression

print("Beta is %s" % beta)

# -------------Graph Solution------------
# graphing results

# pick your favorite!

x_1 = []
y_1 = []
reg_color = []
num = 0  # number of fail to predict

for i in range(n):
    x_1.append(m[i][1])
    y_1.append(m[i][2])
    y_hat = beta[0, 0] + m[i][1] * beta[1, 0] + m[i][2] * beta[2, 0]
    if y_hat > 0.5:
        y_hat = 1.0
    else:
        y_hat = 0.0
    if y_hat != y[i][0]:
        reg_color.append("black")  # fail to predict
        num += 1
    elif y[i][0] == 1.0:
        reg_color.append("skyblue")  # Change Party
    elif y[i][0] == 0.0:
        reg_color.append("pink")  # Traditions Party

print("Number of inaccurate results in Regression is %i" % num + " out of %i" % n)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_1, y_1, c=reg_color)
ax.set_title('Regression')
ax.grid(False)
x_2 = [-40, 60]
y_2 = [78.48969887, 121.1822099]  # solved by making hat = 0.5
ax.plot(x_2, y_2)
plt.xlabel('Number of inaccurate results (black points) is %i' % num + ' out of %i' % n)
plt.show()

# ________________K-NN_____________________
# Here, we do k-nearest neighbor algorithm
# first, we determine our area and step-size

# ------------Mesh over domain------------

step = 2.0  # 3.0 / 5.0

Max = np.squeeze(np.asarray(M.max(0)))  # max vals of columns
Min = np.squeeze(np.asarray(M.min(0)))  # min vals of columns

xmax, ymax = Max[1], Max[2]
xmin, ymin = Min[1], Min[2]

n_x, n_y = int(math.floor((xmax - xmin) / step)), int(math.floor((ymax - ymin) / step))

x_mesh, y_mesh = [], []

xx, yy = xmin, ymin

for i in range(n_x):
    xx = xx + step
    x_mesh.append(xx)

for j in range(n_y):
    yy = yy + step
    y_mesh.append(yy)

# ------------Classify Mesh------------

num_changing_k = []

minK, maxK = 1, 20  # 1, 20

etc = int(round(n_x * n_y / 750) * (maxK - minK))  # an estimate for time

print("The kNN mesh classification may take about %d seconds" % etc)

for k in range(minK, maxK):  # number of neighbors, originally 10

    answer = []

    for i in range(len(x_mesh)):
        for j in range(len(y_mesh)):
            distances = []
            k_ = k
            for l in range(n):
                d = dist(x_mesh[i], y_mesh[j], structured_data[l][0], structured_data[l][1])  # dist1 dist2 dist3
                distances.append([d, structured_data[l][2]])
            distances.sort(key=lambda x_: x_[0])  # sorts by distance, not category
            while distances[k_][0] == distances[k_+1][0] and k_ < len(distances):
                k_ += 1
            topK = distances[0:k_]  # takes top k_ ---> k is not OK with k-th and k+1-th tie.
            cat = mean_cat(topK)
            answer.append([x_mesh[i], y_mesh[j], cat])

    # ------------Print Solution------------
    # our coefficients for linear regression

    # print("kNN mesh categories are %s" % answer)

    # -------------Graph Solution------------
    # graphing results

    # pick your favorite!

    reg_color = []
    num = 0  # number of fail to predict

    for i in range(n):
        x_i = m[i][1]
        y_i = m[i][2]
        # find the position of y_hat
        y_hat = answer[len(answer) - 1][2]
        for j in range(len(answer)):
            if answer[j][0] > x_i and answer[j][1] > y_i:  # We can do this because answer is in ascending order
                y_hat = answer[j][2]
                break
        if y_hat != y[i][0]:
            reg_color.append("black")  # fail to predict
            num += 1
        elif y[i][0] == 1:
            reg_color.append("skyblue")  # Change Party
        elif y[i][0] == 0:
            reg_color.append("pink")  # Traditions Party

    num_changing_k.append([k, num])
    print("Numbers of inaccurate results in kNN mesh categories is %i" % num + " out of %i" % n + " for k = %i" % k)

    if k == 10:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x_1, y_1, c=reg_color)
        ax.set_title('kNN')
        ax.grid(False)
        plt.xlabel('Number of inaccurate results (black points) is %i' % num + ' out of %i' % n + ' for k = %i' % k)
        plt.show()

# find optimal k by minimize inaccurate number

num_changing_k.sort(key=lambda x_: x_[1])
print("Numbers of inaccurate results in kNN mesh categories for step size %.1f are %s " % (step, num_changing_k))
