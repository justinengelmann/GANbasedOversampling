import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score, average_precision_score, \
    mean_squared_error, r2_score, brier_score_loss
from sklearn.utils import resample
from scipy.stats import pearsonr

import random
import string
import datetime

from typing import Tuple

import logging


def make_scores_dict(X_train, y_train, X_test, y_test, clf_list, print_scores=True, bootstrap=True) -> dict:
    metrics_to_use = ['auc', 'brier', 'f1', 'aps', 'acc', 'gmean', 'bacc']

    scores_dict = {'imb': y_train.mean()}

    for name, clf in clf_list:

        clf.fit(X_train, y_train.reshape(-1))
        try:
            preds = clf.predict_proba(X_test)[:, 1]
        except IndexError:
            logging.warning(f'Preds from {name} failed, only one class present.')
            preds = np.array([0] * X_test.shape[0])

        if name == 'rfc':
            if print_scores:
                print(f'Imb: {scores_dict["imb"]:.3f} ', end='')
            _scores = get_scores(y_true=y_test, y_pred=preds, metrics_to_use=metrics_to_use,
                                 bootstrap=bootstrap, print_scores=print_scores)
            scores_dict.update({metric: _scores[metric] for metric in _scores.keys()})
        else:
            _scores = get_scores(y_true=y_test, y_pred=preds, metrics_to_use=metrics_to_use,
                                 bootstrap=bootstrap, print_scores=False)

        scores_dict.update({metric + f'_{name}': _scores[metric] for metric in _scores.keys()})

    # if print_scores:
    #     print(f'Imb: {y_train.mean():.3f} ', end='')
    # # RFC
    # rfc.fit(X_train, y_train.reshape(-1))
    # try:
    #     preds_rfc = rfc.predict_proba(X_test)[:, 1]
    # except IndexError:
    #     logging.warning('Preds from RandomForest failed, only one class present.')
    #     preds_rfc = np.array([0] * X_test.shape[0])
    # scores_dict = get_scores(y_true=y_test, y_pred=preds_rfc, metrics_to_use=metrics_to_use,
    #                          bootstrap=bootstrap, print_scores=print_scores)
    # # LOGIT
    # logit.fit(X_train, y_train.reshape(-1))
    # try:
    #     preds_logit = logit.predict_proba(X_test)[:, 1]
    # except IndexError:
    #     logging.warning('Logit also failed.')
    #     preds_logit = np.array([0] * X_test.shape[0])
    # scores_dict_logit = get_scores(y_true=y_test, y_pred=preds_logit, metrics_to_use=metrics_to_use,
    #                                bootstrap=bootstrap, print_scores=False)
    #
    # scores_dict = {metric: scores_dict[metric] for metric in scores_dict.keys()}
    # scores_dict.update({metric + '_logit': scores_dict_logit[metric] for metric in scores_dict_logit.keys()})
    # scores_dict.update({'imb': y_train.mean()})

    return scores_dict


def generate_date_prefix(random_letters=True) -> str:
    out = f'{str(datetime.date.today())}_{datetime.datetime.now().hour}-{datetime.datetime.now().minute}'
    if random_letters:
        out = f'{out}_{"".join(random.choices(string.ascii_uppercase, k=4))}'
    return out


def get_scores(y_true: np.array, y_pred: np.array,
               metrics_to_use=None,
               threshold: float = 0.5,
               bootstrap: bool = False,
               print_scores: bool = False) -> dict:
    """

    Parameters
    ----------
    y_true
    y_pred
    threshold: float
        threshold for turning scores into class labels
    print_scores: bool
        whether to print the scores

    Returns
    -------
    scores: dict
        dict of metric name: value
    """
    scores = {}
    if metrics_to_use is None:
        metrics_to_use = ['auc', 'aps', 'acc', 'f1', 'bacc', 'brier', 'gmean']

    if not bootstrap:
        # metrics that take predicted probabilities
        for name, metric in [('auc', roc_auc_score), ('aps', average_precision_score), ('brier', brier_score_loss)]:
            if name in metrics_to_use:
                scores[name] = metric(y_true, y_pred)

        # metrics that take class labels
        y_pred_bin = np.where(y_pred > threshold, 1, 0)
        for name, metric in [('acc', accuracy_score), ('bacc', balanced_accuracy_score), ('gmean', geometric_mean_score)]:
            if name in metrics_to_use:
                scores[name] = metric(y_true, y_pred_bin)
        scores['f1'] = f1_score(y_true, y_pred_bin, zero_division=0)
    else:
        scores_lists = {metric: [] for metric in metrics_to_use}
        for i in range(100):
            y_true_res, y_pred_res = resample(y_true, y_pred)
            # recursively get scores
            _scores = get_scores(y_true=y_true_res, y_pred=y_pred_res, metrics_to_use=metrics_to_use)
            for metric in metrics_to_use:
                scores_lists[metric].append(_scores[metric])

        scores = {metric: np.mean(scores_lists[metric]) for metric in metrics_to_use}

    if print_scores:
        for metric in scores.keys():
            print(f'{metric.upper()}:{scores[metric]:.4f} ', end='')
        print()

    return scores


def score_oversampling_performance(X_y_real: torch.Tensor, X_y_fake: torch.Tensor, y_real=None, y_fake=None,
                                   classifier: str = 'rfc') -> dict:
    if y_real is None:
        # assume we are in uncoditional mode and the last two columns of X_y are y as onehot ([0,1] or [1,0])
        # TODO write a better catch (for the opposite case too) or remove
        assert y_fake is None, 'score_oversampling_performance got y_real but no y_fake. Provide neither or both.'
        X_fake = X_y_fake[:, :-2]
        y_fake = X_y_fake[:, -1]
        y_fake = np.where(y_fake > 0.5, 1, 0)
        X_real = X_y_real[:, :-2]
        y_real = X_y_real[:, -1]
    else:
        # assume we are in conditional mode
        X_fake = X_y_fake
        X_real = X_y_real
        y_real = torch.Tensor(y_real).view(-1) if not isinstance(y_real, torch.Tensor) else y_real.view(-1)
        y_fake = torch.Tensor(y_fake).view(-1) if not isinstance(y_fake, torch.Tensor) else y_fake.view(-1)

    X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.1, stratify=y_real)

    # only fake minority data added is class 1
    comb_X = np.vstack([X_train, X_fake[y_fake == 1]])
    comb_y = np.hstack([y_train, y_fake[y_fake == 1]])
    rfc = RandomForestClassifier(n_jobs=6, min_samples_split=10, max_depth=12)
    rfc.fit(comb_X, comb_y)
    comb_preds = rfc.predict_proba(X_test)[:, 1]
    comb_scores = get_scores(y_test, comb_preds)

    # training on only fake data
    rfc = RandomForestClassifier(n_jobs=6, min_samples_split=10, max_depth=12)
    rfc.fit(X_fake, y_fake)
    try:
        fakeonly_preds = rfc.predict_proba(X_test)[:, 1]
    except IndexError:
        logging.warning('Fakeonly preds failed, only one class present.')
        fakeonly_preds = np.array([0] * X_test.shape[0])
    fakeonly_scores = get_scores(y_test, fakeonly_preds)

    return comb_scores, fakeonly_scores


def score_real_fake(X_real: np.array, X_fake: np.array,
                    classifier: str = 'rfc') -> dict:
    rfX = np.vstack([X_real, X_fake])
    rfy = np.hstack([[1] * X_real.shape[0], [0] * X_fake.shape[0]])

    rfX_train, rfX_test, rfy_train, rfy_test = train_test_split(rfX, rfy, test_size=0.2, stratify=rfy)

    if classifier == 'logit':
        clf = LogisticRegression(max_iter=1e4, n_jobs=6)
    elif classifier == 'rfc':
        clf = RandomForestClassifier(max_depth=6, n_estimators=16, min_samples_split=100, n_jobs=6)
    elif classifier == 'rfc_shallow':
        clf = RandomForestClassifier(max_depth=2, n_estimators=16,
                                     min_samples_split=100, max_features=0.1,
                                     n_jobs=6)
    else:
        raise ValueError(f'Unknown classifier "{classifier}". Try one of "logit", "rfc", "rfc_shallow".')

    clf.fit(rfX_train, rfy_train)
    preds = clf.predict_proba(rfX_test)[:, 1]

    scores = get_scores(rfy_test, preds, print_scores=False)

    return scores


def get_dimwise_prob_metrics(X_real: np.array, X_fake: np.array,
                             y_real: np.array = None, y_fake: np.array = None,
                             measure='mean', n_num_cols: int = 0):
    if measure in ['mean', 'avg']:
        real = X_real.mean(axis=0)
        fake = X_fake.mean(axis=0)
    elif measure == 'std':
        real = X_real.std(axis=0)
        fake = X_fake.std(axis=0)
    else:
        raise ValueError(f'"measure" must be "mean" or "std" but "{measure}" was specified.')

    corr_value = pearsonr(real, fake)[0]
    rmse_value = np.sqrt(mean_squared_error(real, fake))

    if n_num_cols > 0:
        num_corr_value = pearsonr(real[:n_num_cols], fake[:n_num_cols])[0]
        num_rmse_value = np.sqrt(mean_squared_error(real[:n_num_cols], fake[:n_num_cols]))
    else:
        num_rmse_value, num_corr_value = -1, -1

    if X_real.shape[1] - n_num_cols > 0:
        cat_corr_value = pearsonr(real[n_num_cols:], fake[n_num_cols:])[0]
        cat_rmse_value = np.sqrt(mean_squared_error(real[n_num_cols:], fake[n_num_cols:]))
    else:
        cat_rmse_value, cat_corr_value = -1, -1,
    return rmse_value, corr_value, num_rmse_value, num_corr_value, cat_rmse_value, cat_corr_value


def make_num_dist_plots(X_real: np.array, X_fake: np.array,
                        y_real: np.array = None, y_fake: np.array = None,
                        show: bool = True, shape: tuple = None, subsample: bool = True,
                        num_cols=None):
    """
    Takes two arrays and plots dimension-wise kdeplots for both arrays.
    Parameters
    ----------
    X_real
    X_fake
    y_real
    y_fake
    show
    shape
    subsample

    Returns
    -------

    """
    if shape is None:
        # by default, we plot 3 columns with up to 2 rows
        if num_cols is not None:
            rows = np.minimum(len(num_cols) // 3, 2)
        else:
            rows = np.minimum(X_real.shape[1] // 3, 2)

        if rows == 0:
            shape = (1, 1)
        else:
            shape = (rows, 3)

    if subsample:
        real_size = int(np.minimum(X_real.shape[0], 5e4))
        fake_size = int(np.minimum(X_fake.shape[0], 5e4))
    else:
        real_size = X_real.shape[0]
        fake_size = X_fake.shape[0]

    fig, axes = plt.subplots(nrows=shape[0], ncols=shape[1])
    fig.set_size_inches((8, 2.25 * shape[0]))
    # print(fig.get_figwidth(),'< width || height >', fig.get_figheight())

    for idx, ax in enumerate(axes.flatten()):
        sns.kdeplot(X_real[:real_size, idx], label='real', ax=ax, shade=True, legend=False, bw=0.02)
        sns.kdeplot(X_fake[:fake_size, idx], label='fake', ax=ax, shade=True, legend=False, bw=0.02)
        ax.set_yticks([])
        ax.set_xticks([0, 1])
        if num_cols is not None:
            ax.set_xlabel(num_cols[idx], labelpad=-10)
    axes.flatten()[0].legend()
    plt.tight_layout()

    if show:
        plt.show()


def make_cat_dist_plots(X_real: np.array, X_fake: np.array,
                        ohe,
                        num_cols: list, cat_cols: list,
                        y_real: np.array = None, y_fake: np.array = None,
                        show: bool = True, shape: tuple = None,
                        log_counts:bool=True):
    if shape is None:
        if len(cat_cols) == 8:
            shape = (2, 4)
        elif len(cat_cols) >= 6:
            shape = (2, 3)
        elif len(cat_cols) >= 3:
            shape = (1, 3)
        elif len(cat_cols) == 2:
            shape = (1, 2)
        else:
            shape = (1, 1)
    end_idx = sum([len(c) for c in ohe.categories_]) + len(num_cols)
    X_fake_cat = pd.DataFrame(ohe.inverse_transform(X_fake[:5000, len(num_cols):end_idx]))
    X_fake_cat['type'] = 'fake'
    X_real_cat = pd.DataFrame(ohe.inverse_transform(X_real[:5000, len(num_cols):end_idx]))
    X_real_cat['type'] = 'real'
    X_real_fake_cat = pd.concat([X_real_cat, X_fake_cat])
    X_real_fake_cat.columns = cat_cols + ['type']

    fig, axes = plt.subplots(shape[0], shape[1])
    fig.set_size_inches((4 * shape[0], 1.12 * shape[1]))
    # print(fig.get_figwidth(),'< width || height >', fig.get_figheight())

    for idx, ax in enumerate(axes.flatten()):
        _plot = sns.countplot(x=cat_cols[idx], hue='type',
                              data=X_real_fake_cat, ax=ax,
                              order=X_real_cat.iloc[:, idx].value_counts().index)
        if idx > 0:
            ax.get_legend().remove()
        else:
            ax.get_legend().remove()
            ax.legend(loc=1)
            ax.get_legend().set_title(None)
        if log_counts:
           _plot.set_yscale("log")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.minorticks_off()
        ax.set_ylabel(None)
    plt.tight_layout()
    if show:
        plt.show()


def make_dimwise_probability_plot(X_real: np.array, X_fake: np.array,
                                  y_real: np.array = None, y_fake: np.array = None,
                                  measure='mean',
                                  show=True, make_fig=True, ax=None,
                                  show_rmse=True, show_corr=True) -> Tuple[float, float]:
    """
    Takes two arrays and plots a scatter plot of a measure (i.e.) the mean for each column of the two arrays against
    each other. The name comes from Bernoulli success probabilities for binary variables, i.e. their mean,
    but this approach generalises to numerical columns. All variables are assumed to be scaled to [0,1].

    Note: Since metrics are computed column-wise, a onehot-encoded column of k-cardinality has k times the impact of a
    single numerical column. Thus it might be wise to compute metrics for both kinds of columns separately.

    Reference: Choi et al., 2017
    Parameters
    ----------
    X_real: np.array
        Array of real data
    X_fake: np.array
        Array of synthetic data
    y_real
    y_fake
    measure: str
        Which measure to plot. Options are ['mean', 'std'].
    show: bool
        Whether to call plt.show()
    make_fig: bool
        Whether to create a new plt figure
    ax:
        plt axes object to plot on
    show_rmse: bool
        Whether to add rmse to the plot
    show_corr: bool
        Whether to add pearson corr coeff to the plot

    Returns
    -------
    rmse_value: float
        root mean square error between the vectors of dimension-wise measure for both arrays.
    corr_value: float
        pearson correlation coefficient between the vectors of dimension-wise measure for both arrays.
    """

    if make_fig and ax is None:
        fig, ax = plt.subplots(1)

    if measure in ['mean', 'avg']:
        real = X_real.mean(axis=0)
        fake = X_fake.mean(axis=0)
    elif measure == 'std':
        real = X_real.std(axis=0)
        fake = X_fake.std(axis=0)
    else:
        raise ValueError(f'"measure" must be "mean" or "std" but "{measure}" was specified.')

    upper_bound = np.maximum(np.max(real) * 1.1, np.max(fake) * 1.1)
    upper_bound = np.minimum(1, upper_bound)

    if measure in ['mean', 'avg']:
        upper_bound = 1
    else:
        upper_bound = 0.6

    ax.scatter(x=real, y=fake)
    ax.plot([0, 1, 2], linestyle='--', c='black')
    ax.set_xlabel('Real')
    ax.set_ylabel('Fake')
    ax.set_xlim(left=0, right=upper_bound)
    ax.set_ylim(bottom=0, top=upper_bound)

    corr_value = pearsonr(real, fake)[0]
    rmse_value = np.sqrt(mean_squared_error(real, fake))

    s = ""
    if show_rmse:
        s += f'RMSE: {rmse_value:.4f}\n'
    if show_corr:
        s += f'CORR: {corr_value:.4f}\n'
    if s != "":
        ax.text(x=upper_bound * 0.98, y=0,
                s=s,
                fontsize=12,
                horizontalalignment='right',
                verticalalignment='bottom')

    if show:
        plt.show()

    return rmse_value, corr_value


def make_dimwise_prediction_performance_plot(X_real: np.array, X_fake: np.array,
                                             y_real: np.array = None, y_fake: np.array = None,
                                             n_dims_to_plot: int = 0,
                                             n_num_cols: int = None,
                                             cat_input_dims: list = None,
                                             show=True, make_fig=True, ax=None,
                                             show_rmse=True, show_corr=True) -> Tuple[float, float]:
    """

    Parameters
    ----------
    X_real: np.array
        Array of real data
    X_fake: np.array
        Array of synthetic data
    y_real
    y_fake
    n_num_cols: int
        number of numerical columns, assumed to be come first

    n_dims_to_plot: int
        the first 'n_dims_to_plot' columns will be plotted. All columns will be used for model fitting
    show: bool
        Whether to call plt.show()
    make_fig: bool
        Whether to create a new plt figure
    ax:
        plt axes object to plot on
    show_rmse: bool
        Whether to add rmse to the plot
    show_corr: bool
        Whether to add pearson corr coeff to the plot

    Returns
    -------
    rmse_value: float
        root mean square error between the vectors of dimension-wise measure for both arrays.
    corr_value: float
        pearson correlation coefficient between the vectors of dimension-wise measure for both arrays.

    """
    # TODO allow to pass saved values for X_real, since we only need to compute it once during training

    if make_fig and ax is None:
        fig, ax = plt.subplots(1)

    if n_num_cols is None:
        n_num_cols = X_real.shape[1]
    if n_dims_to_plot == 0:
        n_dims_to_plot = n_num_cols if cat_input_dims is None else n_num_cols + len(cat_input_dims)

    real, fake = [], []
    for idx in range(n_dims_to_plot):
        for results_list, arr in [(real, X_real), (fake, X_fake)]:
            # Linear regression when using numerical columns as target
            # we use ridge to lessen the need for preprocessing
            if idx < n_num_cols:
                X = arr.copy()[:, ~np.eye(X_real.shape[1])[idx].astype(bool)]
                y = arr.copy()[:, [idx]]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
                model = Ridge()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results_list.append(r2_score(y_test, preds))
            # RandomForest when using categorical columns as target
            else:
                # get cardinality of target
                n_classes = cat_input_dims[idx - n_num_cols]
                start_idx = n_num_cols + sum(cat_input_dims[:idx - n_num_cols])
                end_idx = start_idx + n_classes
                X = np.delete(arr, np.arange(start_idx, end_idx), axis=1)
                y = arr.copy()[:, start_idx: end_idx]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
                model = RandomForestClassifier(n_estimators=20, min_samples_split=0.1, max_depth=6, n_jobs=6)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results_list.append(f1_score(y_test, preds, average='weighted', zero_division=0))

    upper_bound = np.maximum(np.max(real) * 1.1, np.max(fake) * 1.1)
    upper_bound = np.minimum(1, upper_bound)

    ax.scatter(x=real, y=fake)
    ax.plot([0, 1], linestyle='--', c='black')
    ax.set_xlabel('Real')
    ax.set_ylabel('Fake')
    ax.set_xlim(left=0, right=upper_bound)
    ax.set_ylim(bottom=0, top=upper_bound)

    corr_value = pearsonr(real, fake)[0]
    rmse_value = np.sqrt(mean_squared_error(real, fake))

    s = ""
    if show_rmse:
        s += f'RMSE: {rmse_value:.4f}\n'
    if show_corr:
        s += f'CORR: {corr_value:.4f}\n'
    if s != "":
        ax.text(x=upper_bound * 0.98, y=0,
                s=s,
                fontsize=12,
                horizontalalignment='right',
                verticalalignment='bottom')

    if show:
        plt.show()

    return rmse_value, corr_value


def save_current_plot(name: str, prefix: str = '', path='', show=False, clear=False):
    filename = f'{path}/{prefix}{name}.pdf'
    plt.savefig(filename, dpi=100)
    if show:
        # TODO clean up. hacky solution to surpress plots when not developing
        # plt.clf()
        # plt.close()
        plt.show()
    if clear:
        plt.clf()


def get_cat_dims(X, cat_cols) -> list:
    """
    Takes a pd.DataFrame and a list of columns and returns a list of levels/cardinality per column in the same order.
    """
    return [(X[col].nunique()) for col in cat_cols]
