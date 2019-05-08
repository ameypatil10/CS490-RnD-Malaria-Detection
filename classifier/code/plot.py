import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_train(csv_file):
    data = pd.read_csv(csv_file, header=None)
    data = data.T
    data = pd.DataFrame(data.values, columns = ["training loss", "Precision", "Recall", "F1 score", "MCC", "Training Accuracy"])
    data.drop(data.tail(1).index,inplace=True)
    training_loss = data[["training loss"]].plot(kind='line')
    plt.savefig('../results/training_loss.png')
    # plt.show()
    scores = data[["Precision", "Recall", "F1 score", "MCC"]].plot(kind='line')
    plt.savefig('../results/training_scores.png')
    # plt.show()
    acc = data[["Training Accuracy"]].plot(kind='line')
    plt.savefig('../results/training_accuracy.png')
    # plt.show()

def plot_snap(csv_file):
    data = pd.read_csv(csv_file, header=None)
    data = data.T
    data = pd.DataFrame(data.values, columns = ["Training loss", "Training Precision", "Training Recall", "Training F1 score", "Training MCC", "Training Accuracy", "Validation Precision", "Validation Recall", "Validation F1 score", "Validation MCC", "Validation Accuracy"])
    # data.columns = ["training loss", "Precision", "Recall", "F1 score", "MCC", "Accuracy"]
    data.drop(data.tail(1).index,inplace=True)
    acc = data[["Training Accuracy", "Validation Accuracy"]].plot(kind='line')
    plt.savefig('../results/accuracy.png')
    # plt.show()
    prec_rec = data[["Training Precision", "Training Recall", "Validation Precision", "Validation Recall"]].plot(kind='line')
    plt.savefig('../results/prec_recall.png')
    # plt.show()
    F1 = data[["Training F1 score", "Validation F1 score"]].plot(kind='line')
    plt.savefig('../results/F1.png')
    # plt.show()
    MCC = data[["Training MCC", "Validation MCC"]].plot(kind='line')
    plt.savefig('../results/MCC.png')
    # plt.show()

plot_train('../results/train_log.csv')

plot_snap('../results/snap_log.csv')