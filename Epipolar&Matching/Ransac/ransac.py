# https://salzis.wordpress.com/2014/06/10/robust-linear-model-estimation-using-ransac-python-implementation/
import math
import sys

import numpy as np
import scipy
from Ransac.ransac_util import ransac_plot

# Ransca parameters
ransac_iterations = 20
ransac_threshold = 3
ransac_ratio = 0.6 # Ratio of inliers required to assert that a model fits well to data.

# generate sparse input data
n_samples = 500
outliers_ratio = 0.4  # ratio of outliers

n_inputs = 1
n_outputs = 1

# generate samples
x = 30 * np.random.random((n_samples, n_inputs))

# generate line's slope (called here perfect fit)
perfect_fit = 0.5 * np.random.normal(size=(n_inputs, n_outputs))

# compute output
y = scipy.dot(x, perfect_fit)

# add a little gaussian noise
x_noise = x + np.random.normal(size=x.shape)
y_noise = y + np.random.normal(size=y.shape)

# add some outliers to the point-set
n_outliers = outliers_ratio * n_samples
indices = np.arange(x_noise.shape[0])
np.random.shuffle(indices)
outlier_indices = indices[: int(n_outliers)]

x_noise[outlier_indices] = 30 * np.random.random(size=(n_outliers, n_inputs))

# gaussian outliers
y_noise[outlier_indices] = 30 * np.random.normal(size=(n_outliers, n_outputs))

# non-gaussian outliers (only on one side)
# y_noise[outlier_indices] = 30*(np.random.normal(size=(n_outliers,n_outputs))**2)

def find_line_model(points):
    """ find a line model for the given points
    :param points selected points for model fitting
    :return line model
    """

    # [WARNING] vertical and horizontal lines should be treated differently
    #           here we just add some noise to avoid division by zero

    # find a line model for these points
    m = (points[1, 1] - points[0, 1]) / ( points[1, 0] - points[0, 0] + sys.float_info.epsilon)  # slope (gradient) of the line
    c = points[1, 1] - m * points[1, 0]  # y-intercept of the line

    return m, c


def find_intercept_point(m, c, x0, y0):
    """ find an intercept point of the line model with
        a normal from point (x0,y0) to it
    :param m slope of the line model
    :param c y-intercept of the line model
    :param x0 point's x coordinate
    :param y0 point's y coordinate
    :return intercept point
    """

    # intersection point with the model
    x = (x0 + m * y0 - m * c) / (1 + m ** 2)
    y = (m * x0 + (m ** 2) * y0 - (m ** 2) * c) / (1 + m ** 2) + c

    return x, y


data = np.hstack((x_noise, y_noise))

ratio = 0.
model_m = 0.
model_c = 0.

# perform RANSAC iterations
for it in range(ransac_iterations):
    # pick up two random points
    n = 2

    all_indices = np.arange(x_noise.shape[0])
    np.random.shuffle(all_indices)

    indices_1 = all_indices[:n]
    indices_2 = all_indices[n:]

    maybe_points = data[indices_1, :]
    test_points = data[indices_2, :]

    # find a line model for these points
    m, c = find_line_model(maybe_points)

    x_list = []
    y_list = []
    num = 0

     # find orthogonal lines to the model for all testing points
    for ind in range(test_points.shape[0]):

        x0 = test_points[ind, 0]
        y0 = test_points[ind, 1]

        # find an intercept point of the model with a normal from point (x0,y0)
        x1, y1 = find_intercept_point(m, c, x0, y0)

        # distance from point to the model
        dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        # check whether it's an inlier or not
        if dist < ransac_threshold:
            x_list.append(x0)
            y_list.append(y0)
            num += 1

    x_inliers = np.array(x_list)
    y_inliers = np.array(y_list)

    # in case a new model is better - cache it
    if num / float(n_samples) > ratio:
        ratio = num / float(n_samples)
        model_m = m
        model_c = c

    print('inlier ratio = ', num / float(n_samples))
    print('  model_m = ', model_m)
    print('  model_c = ', model_c)

    # plot the current step
    ransac_plot(it, x_noise, y_noise, m, c, False, x_inliers, y_inliers, maybe_points)

    # we are done in case we have enough inliers
    if num > n_samples * ransac_ratio:
        print('The model is found !')
        break

# plot the final model
ransac_plot(0, x_noise, y_noise, model_m, model_c, True)

print('\nFinal model:\n')
print('  ratio = ', ratio)
print('  model_m = ', model_m)
print('  model_c = ', model_c)