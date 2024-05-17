import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression
from p05b_lwr import fit_and_plot


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    mse = []
    for tau in tau_values:
        tmp = fit_and_plot(tau, x_train, y_train, x_valid, y_valid, f'output/p05c_{tau:.2f}.png')
        mse.append(tmp)

    for mse_i in mse:
        print(f'MSE: {mse_i}')
    min_tau = tau_values[np.argmin(mse)]
    print(f'min tau of MSE: {min_tau}')
    model = LocallyWeightedLinearRegression(tau=min_tau)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    np.savetxt(pred_path, y_pred)

    # *** END CODE HERE ***
