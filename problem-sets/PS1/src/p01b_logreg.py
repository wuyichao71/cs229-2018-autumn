import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # print(model.theta)
    util.plot(x_train, y_train, model.theta, save_path=f'output/p01b_{pred_path[-5]}')
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    # np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        # g(z)
        def g(theta, x):
            return 1 / (1 + np.exp(-x.dot(theta)))

        # H
        def hessian(theta, x):
            h_theta = g(theta, x)
            return (x.T * h_theta * (1 - h_theta)).dot(x) / m

        # gradient
        def grad(theta, x, y):
            m, n = x.shape
            return x.T.dot(g(theta, x) - y) / m

        # update theta
        def update(theta, x, y):
            return theta - np.linalg.inv(hessian(theta, x)).dot(grad(theta, x, y))

        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)

        theta = self.theta
        for i in range(self.max_iter):
        # while True:
            theta_old = theta
            theta = update(theta, x, y)
            # print(theta)
            if np.linalg.norm(theta - theta_old, 1) < self.eps:
                break
        self.theta = theta

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE ***
