import matplotlib.pyplot as plt

def ransac_plot(n, x, y, m, c, final=False, x_in=(), y_in=(), points=()):
    """ plot the current RANSAC step
    :param n      iteration
    :param points picked up points for modeling
    :param x      samples x
    :param y      samples y
    :param m      slope of the line model
    :param c      shift of the line model
    :param x_in   inliers x
    :param y_in   inliers y
    """

    fname = "output/figure_" + str(n) + ".png"
    line_width = 1.
    line_color = '#0080ff'
    title = 'iteration ' + str(n)

    if final:
        fname = "output/final.png"
        line_width = 3.
        line_color = '#ff0000'
        title = 'final solution'

    plt.figure("Ransac", figsize=(15., 15.))

    # grid for the plot
    grid = [min(x) - 10, max(x) + 10, min(y) - 20, max(y) + 20]
    plt.axis(grid)

    # put grid on the plot
    plt.grid(b=True, which='major', color='0.75', linestyle='--')
    plt.xticks([i for i in range(min(x) - 10, max(x) + 10, 5)])
    plt.yticks([i for i in range(min(y) - 20, max(y) + 20, 10)])

    # plot input points
    plt.plot(x[:, 0], y[:, 0], marker='o', label='Input points', color='#00cc00', linestyle='None', alpha=0.4)

    # draw the current model
    plt.plot(x, m * x + c, 'r', label='Line model', color=line_color, linewidth=line_width)

    # draw inliers
    if not final:
        plt.plot(x_in, y_in, marker='o', label='Inliers', linestyle='None', color='#ff0000', alpha=0.6)

    # draw points picked up for the modeling
    if not final:
        plt.plot(points[:, 0], points[:, 1], marker='o', label='Picked points', color='#0000cc', linestyle='None',
                 alpha=0.6)

    plt.title(title)
    plt.legend()
    plt.savefig(fname)
    plt.close()