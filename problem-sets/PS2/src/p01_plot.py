import util
import numpy as np
import matplotlib.pyplot as plt
import os


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    print(probs)
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad

def prediction(X, theta):
    m, n = X.shape
    Y = np.zeros(m)
    border = X.dot(theta)
    Y[border >= 0] = 1
    Y[border < 0] = -1
    return Y


def logistic_regression(X, Y, niter=None):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    i = 0
    while niter is None or i < niter:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return theta

def plot(x, y, theta, save_path=None, correction=1.0, label=[1,-1]):
    """Plot dataset and fitted logistic regression parameters.
    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply (Problem 2(e) only).
    """
    # Plot dataset
    plt.figure()
    plt.plot(x[y == label[0], -2], x[y == label[0], -1], 'bx', linewidth=2)
    plt.plot(x[y == label[1], -2], x[y == label[1], -1], 'go', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    margin1 = (max(x[:, -2]) - min(x[:, -2]))*0.2
    margin2 = (max(x[:, -1]) - min(x[:, -1]))*0.2
    if theta is not None:
        x1 = np.arange(min(x[:, -2])-margin1, max(x[:, -2])+margin1, 0.01)
        x2 = -(theta[0] / theta[2] * correction + theta[1] / theta[2] * x1)
        plt.plot(x1, x2, c='red', linewidth=2)
    plt.xlim(x[:, -2].min()-margin1, x[:, -2].max()+margin1)
    plt.ylim(x[:, -1].min()-margin2, x[:, -1].max()+margin2)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    if save_path is not None:
        plt.savefig(save_path)

def main():
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    theta_a = logistic_regression(Xa, Ya)
    Ya_pred = prediction(Xa, theta_a)
    plot(Xa, Ya, theta_a)
    plot(Xa, Ya_pred, theta_a)

    # Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    # theta_b = logistic_regression(Xb, Yb, 100000)
    # Yb_pred = prediction(Xb, theta_b)
    # plot(Xb, Yb, theta_b)
    
    plt.show()


if __name__ == '__main__':
    main()

