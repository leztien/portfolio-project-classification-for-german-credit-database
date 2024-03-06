

"""
Custom made plotting functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from math import sqrt, ceil
from sklearn.metrics import confusion_matrix




def plot_stacked_bar(df, feature):
    table = pd.crosstab(df[feature], df['label']).T
    cats = table.columns

    neg, pos = table.values

    # Convert values to proportions
    total = np.array(neg) + np.array(pos)
    neg = np.array(neg) / total
    pos = np.array(pos) / total

    # Create stacked bar chart
    plt.bar(cats, neg, label="0", color='green')
    plt.bar(cats, pos, bottom=neg, label="1", color='red')

    plt.xticks(range(len(cats)), cats, rotation=-45)
    plt.title(f"Proportion of postitive class per category in '{feature}'")
    plt.legend()


# Function that draws stacked bar chart
def make_stacked_barchart(df, feature, target='label', ax=None):
    """
    Docs: TODO
    """

    ax = ax or plt.subplot()
    
    counts = pd.crosstab(df[target], df[feature])
    counts.columns = counts.columns.astype(str)
    counts = counts[np.take(counts.columns, np.argsort(counts.sum(axis=0).values)[::-1])]
    props = counts / counts.sum().sum()
    
    bottom = np.zeros(counts.shape[1])
    
    for y,p in props.iterrows():
        ax.bar(counts.columns, height=p, bottom=bottom, label=str(y), color=['green', 'red'][int(y)])
        bottom += p
    
    ax.set_title(feature, fontsize=10)
    ax.set_xticks(list(range(counts.shape[1])), counts.columns, 
               rotation=-20, fontsize=8)
    ax.tick_params(axis='y', labelsize=6)

    ax.legend(title="Classes:")
    return ax


def make_stacked_barcharts(df, features):
    """
    A convenience wrapper function
    """

    ncols = ceil(sqrt(len(features)) * 1.2)
    nrows = ceil(len(features) / ncols)
    #ncols = nrows = ceil(sqrt(len(features)))  # alternatevely

    fig, axes = plt.subplots(nrows, ncols, sharey=False, figsize=(12,10))
    fig.suptitle("Proportions of categories and\nthe proportions of the positive class per category\nin a given feature")

    for i, ax in enumerate(axes.flatten()):
        if i >= len(features):
            plt.delaxes(ax)
            continue
        ax = make_stacked_barchart(df, feature=features[i], target='label', ax=ax)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    return fig


def make_barchart(df, feature, target='label', annotations=False, ax=None):
    """
    Makes a bar chart 
    x = categories
    y = proportions
    the categories on the x-axis are sorted by the proportion of bad loans
    """

    ax = ax or plt.subplot()

    data = df[[feature, target]].groupby(feature).agg(['size', 'mean'])

    data.columns = data.columns.droplevel(level=0)
    data['size'] /= data['size'].sum()
    data.columns = ["prop_total", "prop_bad"]
    data = data.sort_values('prop_bad', ascending=False)
    data["color"] = (data['prop_bad'] * 100).astype(int)

    cmap = sns.color_palette('Reds', n_colors=data['color'].max()+1)
    colors = np.array(cmap)[data['color']]

    ax.bar(data.index.astype(str), height=data['prop_total'], color=colors, edgecolor='grey')
    ax.grid(True, which='major', axis='y')
    ax.set_xticks(range(len(data.index)), data.index, rotation=-20, fontsize=7)
    
    ax.set_title(feature)
    #ax.set_ylabel("proporton")
    #ax.set_title(f"Categories in '{feature}' and their proportions\n(sorted by the proportion of 'bad' loans)")

    # Annotate bars with their heights
    if annotations:
        for i, (h, k) in enumerate(zip(data['prop_total'], data['prop_bad'].round(2))):
            ax.text(i, h-0.011, str(k), ha='center', fontsize=9)

    return ax



def make_barcharts(df, features, annotations=True, square=True, figsize=(14,10)) -> plt.Figure:
    """
    A convenience wrapper function which plots multiple bar charts
    """

    if square:
        ncols = nrows = ceil(sqrt(len(features)))
    else:
        ncols = ceil(sqrt(len(features)) * 1.2)
        nrows = ceil(len(features) / ncols)

    fig, axes = plt.subplots(nrows, ncols, sharey=False, figsize=figsize)
    fig.suptitle("Proportions of categories and\nthe proportions of the positive class per category\nin a given feature")

    for i, ax in enumerate(axes.flatten()):
        if i >= len(features):
            plt.delaxes(ax)
            continue
        ax = make_barchart(df, feature=features[i], ax=ax)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    return fig


def plot_probabilities_distributions(model, X, y, threshold=0.5, n_bins=50, alpha=0.5, plot_confusion_matrix=True):
    """
    Make distributions of probabilities for true and false predictions
    """

    ytrue = np.array(y)
    ppred = model.predict_proba(X)[:,-1]
    ypred = (ppred >= threshold).astype(int)
    mask = ytrue == ypred

    height = max([
        max(plt.hist(ppred[mask], bins=n_bins, alpha=alpha, color='green', label="true predictions")[0]),
        max(plt.hist(ppred[~mask], bins=n_bins, alpha=alpha, color='red', label="false predictions")[0])
    ])

    plt.xlim([0,1])
    plt.vlines(threshold, ymin=0, ymax=height/2, color='grey', label="threshold")
    plt.legend()
    plt.title(f"Distribution of probabilities by {model[-1].__class__.__name__}")

    # confusion matrix
    if plot_confusion_matrix:
        ax = plt.axes([0.4, 0.62, 0.2, 0.2])  # [left, bottom, width, height]
        #ypred = cross_val_predict(model, X, ytrue, cv=5)
        cm = confusion_matrix(ytrue, ypred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, linecolor='k')
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('Actual', fontsize=9)
        ax.set_title("Confusion Matrix", fontsize=10)
        ax.set_facecolor('grey')
    return #plt.gca()