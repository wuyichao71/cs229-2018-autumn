import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    def fit(label_col, part, pred_path):
        x_train, y_train = util.load_dataset(train_path, label_col=label_col, add_intercept=True)
        x_test, y_test = util.load_dataset(test_path, label_col=label_col, add_intercept=True)
        model = LogisticRegression()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        util.plot(x_train, y_train, model.theta, save_path=f'output/p02{part}_pred')
        util.plot(x_test, y_test, model.theta, save_path=f'output/p02{part}_test')
        np.savetxt(pred_path, y_pred>0.5, fmt='%d')
        return model

    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    fit('t', 'c', pred_path_c)

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    model = fit('y', 'd', pred_path_d)
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    x_train, y_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    util.plot(x_valid, y_valid, model.theta, save_path=f'output/p02d_valid')
    y_valid_pred = model.predict(x_valid)
    y_pred = model.predict(x_test)
    alpha = y_valid_pred[y_valid == 1].mean()
    correction = 1 + np.log(2 / alpha - 1) / model.theta[0]
    util.plot(x_train, y_train, model.theta, save_path=f'output/p02e_pred', correction=correction)
    util.plot(x_test, y_test, model.theta, save_path=f'output/p02e_test', correction=correction)
    # print(model.theta[0] * correction)
    np.savetxt(pred_path_e, y_pred / alpha>0.5, fmt='%d')
    # *** END CODER HERE
