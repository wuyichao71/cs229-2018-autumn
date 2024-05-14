from p01b_logreg import LogisticRegression
from p01e_gda import GDA
import matplotlib.pyplot as plt
import numpy as np
import util

def fit_and_plot(train_path, eval_path, pred_path):
    x_train, y_train = util.load_dataset(train_path)
    x_train_intercept = util.add_intercept(x_train)

    logreg = LogisticRegression()
    logreg.fit(x_train_intercept, y_train)

    gda = GDA()
    gda.fit(x_train, y_train)

    plot(x_train, y_train, logreg.theta, gda.theta, save_path=f'output/p01_fit_{pred_path[-5]}')

    x_eval, y_eval = util.load_dataset(eval_path)
    x_eval_intercept = util.add_intercept(x_eval)

    logreg.predict(x_eval_intercept)
    gda.predict(x_eval_intercept)

    plot(x_eval, y_eval, logreg.theta, gda.theta, save_path=f'output/p01_pred_{pred_path[-5]}')

    if (int(pred_path[-5]) == 1):
        x_log_train = log_x2(x_train)
        x_log_train_intercept = util.add_intercept(x_log_train)

        logreg = LogisticRegression()
        logreg.fit(x_log_train_intercept, y_train)

        gda = GDA()
        gda.fit(x_log_train, y_train)

        plot(x_log_train, y_train, logreg.theta, gda.theta, save_path=f'output/p01_log_fit_{pred_path[-5]}')

        x_log_eval = log_x2(x_eval)
        x_log_eval_intercept = util.add_intercept(x_log_eval)

        logreg.predict(x_log_eval_intercept)
        gda.predict(x_log_eval_intercept)

        plot(x_log_eval, y_eval, logreg.theta, gda.theta, save_path=f'output/p01_log_pred_{pred_path[-5]}')

def log_x2(x_train):
    x_log_train = np.zeros_like(x_train)
    x_log_train[:] = x_train[:]
    x_log_train[:, -1] = np.log(x_train[:, -1])
    return x_log_train


def plot(x, y, theta_1, theta_2, save_path=None, correction=1.0):
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
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    margin1 = (max(x[:, -2]) - min(x[:, -2]))*0.2
    margin2 = (max(x[:, -1]) - min(x[:, -1]))*0.2
    if theta_1 is not None:
        x1 = np.arange(min(x[:, -2])-margin1, max(x[:, -2])+margin1, 0.01)
        x2 = -(theta_1[0] / theta_1[2] * correction + theta_1[1] / theta_1[2] * x1)
        plt.plot(x1, x2, c='red', linewidth=2)
    if theta_2 is not None:
        x1 = np.arange(min(x[:, -2])-margin1, max(x[:, -2])+margin1, 0.01)
        x2 = -(theta_2[0] / theta_2[2] * correction + theta_2[1] / theta_2[2] * x1)
        plt.plot(x1, x2, c='black', linewidth=2)
    plt.xlim(x[:, -2].min()-margin1, x[:, -2].max()+margin1)
    plt.ylim(x[:, -1].min()-margin2, x[:, -1].max()+margin2)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    if save_path is not None:
        plt.savefig(save_path)

if __name__ == '__main__':
    fit_and_plot(train_path='../data/ds1_train.csv',
         eval_path='../data/ds1_valid.csv',
         pred_path='output/p01b_pred_1.txt')
    fit_and_plot(train_path='../data/ds2_train.csv',
         eval_path='../data/ds2_valid.csv',
         pred_path='output/p01b_pred_2.txt')

