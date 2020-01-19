import os
import numpy as np
from Anomaly_NN import train_NN

anomaly_vid = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
               "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting",
               "Stealing", "Vandalism"]
normal_vid = ["Normal_Videos_event", "Testing_Normal_Videos_Anomaly",
              "Training_Normal_Videos_Anomaly"]


def train_model(train_ano, train_norm, test_ano, test_norm, ano_width, norm_width):
    train_NN(train_ano, train_norm,
             test_ano, test_norm,
             ano_width[125:], norm_width[125:])


def read_csv(filepath):
    width = []
    feature = []
    with open(filepath, 'r') as f:
        width.append(float(f.readline().split("\n")[0]))
        for cnt, line in enumerate(f):
            if cnt == 32:
                break
            line = line.split("\n")[0]
            feature.append(map(float, line.split(",")))

    return width, feature


def read_all_csv(root, folderlist):
    widths = []
    features = []
    for folder in folderlist:
        path = os.path.join(root, folder)
        for file in os.listdir(path):
            filepath = os.path.join(path, file)
            width, feature = read_csv(filepath)
            widths = np.hstack(widths, width)
            features = np.hstack(features, feature)

    return widths, features


ano_data, ano_width = read_all_csv('Train_Video_Feature', anomaly_vid)
normal_data, norm_width = read_all_csv('Train_Video_Feature', normal_vid)
train_ano = ano_data[0:4000]
train_norm = normal_data[0:4000]
test_ano = ano_data[4000:]
test_norm = normal_data[4000:]
train_model(train_ano, train_norm, test_ano, test_norm)
