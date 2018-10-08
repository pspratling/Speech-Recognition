import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

WORDS = os.listdir('../data/train')
COMMAND_WORDS = sorted(['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence'])


#functions for loading in mfcc data
def load_word(fold, word, augmented=False):
    """
    Reads in MFCC data for a single word in a given fold
    fold: 'train', 'validation' or 'test'
    word: word to extract
    augmented: boolean value that determines if the augmented data is loaded along with the given data
    Returns tuple of X and y data for that word
    """
    if augmented and word != 'silence' and fold == 'train':
        try:
            X = np.load('../data/{}/{}/mfcc_augmented.npy'.format(fold, word))
        except:
            X = np.load('../data/{}/{}/mfcc.npy'.format(fold, word))
    else:
        X = np.load('../data/{}/{}/mfcc.npy'.format(fold, word))
    y = np.full(shape=X.shape[0],
                fill_value=word if word in COMMAND_WORDS else 'unknown')
    return X, y


def get_fold_data(fold, words=WORDS, augmented=False):
    """
    Reads in mfcc data  and labels for entire fold, calling load_word
    fold: 'train', 'validation', or 'test'
    words: list of words to extract
    augmented: boolean value that determines if the augmented data is loaded along with the given data
    """
    
    X_total = np.array([]).reshape(0, 20 , 32)
    y_total = np.array([])
    for word in words:
        X_word, y_word = load_word(fold, word, augmented)
        X_total = np.vstack([X_total, X_word])
        y_total = np.append(y_total, y_word)
        
    return X_total, y_total


#functions for reformatting data into forms needed for neural networks
def reformat_X(X):
    """
    Reshapes X into an 'image' of depth 1
    """
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    return X

def reformat_y(y):
    """
    Formats y into one hot encoded integers
    """
    y = LabelEncoder().fit_transform(y)
    y = to_categorical(y)
    return y


#helper functions for evaluating model results
def plot_confusion_matrix(y_obs_labels, y_pred_classes, words=COMMAND_WORDS):
    """
    Returns a confusion matrix showing model performance for each word.
    y_obs_labels: 1xn array with each entry the observed y word as a string
    y_pred_classes: 1xn array with each entry the predicted class of the word as an int
    words: array of words in data set
    """
    y_pred_labels = np.array([words[y] for y in y_pred_classes])
    conf_matrix = confusion_matrix(y_obs_labels, y_pred_labels)
    conf_matrix_percent = np.divide(conf_matrix.T, conf_matrix.sum(axis=1)).T
    fig, ax = plt.subplots(figsize=(12, 12))    
    sns.heatmap(conf_matrix_percent, cmap='Blues', ax=ax, annot=True, fmt='.2f', cbar=False)
    ax.set_xticklabels(sorted(words), rotation=90)
    ax.set_yticklabels(sorted(words), rotation=0)
    ax.set_ylabel('Observed')
    ax.set_xlabel('Predicted')
    plt.show()
    return conf_matrix_percent  


def create_summary_df(y_obs, y_pred_proba, y_pred_classes):
    """
    Takes in Keras model output and returns summary Dataframe detailing model performance
    y_obs: observed y values
    y_pred_proba: predicted model probabilites
    y_pred_classes: predicted model classes
    """
    labels = COMMAND_WORDS + ['unknown']
    df = pd.DataFrame()
    df['y_obs_words'] = y_obs
    df['y_pred_words'] = [labels[class_no] for class_no in y_pred_classes]
    df['max_confidences'] = y_pred_proba.max(axis=1)
    
    def classify_labels(row):
        if row['y_obs_words'] == 'unknown':
            return 'word not in training set'
        elif row['y_obs_words'] == row['y_pred_words']:
            return 'correct'
        else:
            return 'misclassified'
        
    df['labels'] = df.apply(classify_labels, axis=1)
    df['is_correct'] = (df['labels']=='correct')
    return df


def plot_hist(df, ax, color, ylabel, xlabel=False, title=False):
    """
    Plots summary histogram
    df: Dataframe from create_summary_df
    ax: matplotlib axes to plot on
    color: color of histogram
    xlabel: boolean that determines whether to label x axis
    title: boolean that determines whether to label title
    """
    hist = df[['labels', 'max_confidences']].hist(by='labels', color=color, ax=ax, bins=20)
    hist[0].set_ylabel(ylabel)
    for ax in hist:
        ax.set_xticks(np.linspace(0, 1, num=21))
    if not title:
        for ax in hist:
            ax.set_title('')
    if xlabel:
        for ax in hist:
            ax.set_xlabel('maximum classification confidence')          
    return hist
    
    
def plot_hist_grid(y_obs, y_pred_proba, y_pred_classes, title=''):
    """
    Calls plot_hist and plots entire summary histogram grid
    y_obs: observed y values
    y_pred_proba: predicted probabilities for y values
    y_pred_classes: predicted classes of y values
    title: title of plot
    """
    train_df = create_summary_df(y_obs['train'], y_pred_proba['train'], y_pred_classes['train'])
    val_df = create_summary_df(y_obs['val'], y_pred_proba['val'], y_pred_classes['val'])
    
    fig, ax = plt.subplots(2, 3, figsize=(15 , 8))
    plot_hist(train_df, color='C0', title=True, ylabel='train samples', ax=ax[0])
    plot_hist(val_df, color='C1', ylabel='validation samples', xlabel=True, ax=ax[1])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(title)
    return ax


def plot_roc_curve(df, ax):
    """
    Plots roc curve
    df: Dataframe from create_summary_df
    ax: matplotlib axes to plot on
    """
    fpr, tpr, _ = roc_curve(df['is_correct'], df['max_confidences'])

    ax.step(fpr, tpr, color='b', alpha=0.2,
         where='post')
    ax.fill_between(fpr, tpr, step='post', alpha=0.2,
                 color='b')

    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    ax.set_aspect('equal')
    return ax;


def modify_pred_classes(y_pred_proba, y_pred_classes, thresh=.9):
    """
    Modifies predicted classes to account for words not in command words set. Based on prediction confidence.
    y_pred_proba: array of prediction probability output from Keras model
    y_pred_classes: array of prediction classes from Keras model
    thresh: threshold to predict anything less confident than as "unknown"
    """
    probs = y_pred_proba.max(axis=1)
    is_unknown = probs < thresh
    y_pred_modified = y_pred_classes.copy()
    y_pred_modified[is_unknown] = 11
    return y_pred_modified
  
    
def score_with_unknown(y_obs, y_pred_classes, words=COMMAND_WORDS+['unknown']):
    """
    Scores model performance with unknown values
    y_obs: observed y values
    y_pred_classes: predicted classes from Keras model
    words: set of words to classify
    """
    pred_words = np.array([words[y] for y in y_pred_classes])
    accuracy = (pred_words == y_obs).sum()/y_obs.size
    return accuracy