import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel

def fit_and_plot(tau, x_train, y_train, x_eval, y_eval, figname):
    lwr = LocallyWeightedLinearRegression(tau=tau)
    lwr.fit(x_train, y_train)
    y_pred = lwr.predict(x_eval)
    mse = np.mean((y_pred - y_eval) ** 2)
    plt.plot(x_eval[:, -1], y_eval, 'bx', label='label')
    plt.plot(x_eval[:, -1], y_pred, 'ro', label='predict')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(figname)
    plt.clf()
    return mse

def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)


    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    mse = fit_and_plot(tau, x_train, y_train, x_eval, y_eval, 'output/p05b.png')
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # print(x.shape)
        # print(self.x.shape)
        y = []
        for x1 in x:
            w_vector = np.exp(-np.linalg.norm(self.x - x1, ord=2, axis=-1)**2 / (2 * self.tau**2))
            xwx = (self.x.T * w_vector).dot(self.x)
            theta = np.linalg.inv(xwx).dot((self.x.T * w_vector).dot(self.y))
            y.append(x1.dot(theta))
        return np.array(y)
        # *** END CODE HERE ***
