import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_train(csv_file):
    data = pd.read_csv(csv_file, header=None)
    data = data.T
    data = pd.DataFrame(data.values, columns = ["training loss", "training iou", "Training Accuracy"])
    data.drop(data.tail(1).index,inplace=True)
    training_loss = data[["training loss"]].plot(kind='line')
    plt.savefig('../results/training_loss1.png')
    # plt.show()
    iou = data[["training iou"]].plot(kind='line')
    plt.savefig('../results/training_iou1.png')
    # plt.show()
    acc = data[["Training Accuracy"]].plot(kind='line')
    plt.savefig('../results/training_accuracy1.png')
    # plt.show()

def plot_snap(csv_file):
    data = pd.read_csv(csv_file, header=None)
    data = data.T
    data = pd.DataFrame(data.values, columns = ["train_loss", "train_iou", "train_accuracy", "valid_iou", "valid_accuracy"])
    # data.columns = ["training loss", "Precision", "Recall", "F1 score", "MCC", "Accuracy"]
    data.drop(data.tail(1).index,inplace=True)
    acc = data[["train_accuracy", "valid_accuracy"]].plot(kind='line')
    plt.savefig('../results/accuracy1.png')
    # plt.show()
    prec_rec = data[["train_iou", "valid_iou"]].plot(kind='line')
    plt.savefig('../results/iou1.png')
    # plt.show()

plot_train('../results/train_log.csv')

plot_snap('../results/snap_log.csv')
