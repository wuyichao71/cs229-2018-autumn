import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    
    model = PoissonRegression(step_size=lr)
    model.fit(x_train, y_train)

    # util.plot(x_train, y_train, model.theta, save_path='output/p03d_pred')
    # util.plot(x_eval, y_eval, model.theta, save_path='output/p03d_eval')
    y_pred = model.predict(x_eval)
    plt.plot(y_eval, color='r', ls='', marker='o')
    plt.plot(y_pred, color='g', ls='', marker='x')
    plt.show()

    np.savetxt(pred_path, y_pred, fmt='%d')
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape

        def h(theta, x):
            return np.exp(x.dot(theta))

        def gradient(theta, x, y):
            return (y - h(theta, x)).dot(x) / m

        def update_theta(theta, x, y):
            return theta + self.step_size * gradient(theta, x, y)

        if self.theta is None:
            self.theta = np.zeros(n)

        theta = self.theta
        # for i in range(self.max_iter):
        while True:
            old_theta = theta
            theta = update_theta(theta, x, y)
            if np.linalg.norm(theta - old_theta, 1) < self.eps:
                break
        self.theta = theta
        print(self.theta)

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x.dot(self.theta))
        # *** END CODE HERE ***
