from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adagrad
from scipy.io import loadmat, savemat
from keras.models import model_from_json
import theano.tensor as T
import theano
import os
from os import listdir
import numpy as np
from datetime import datetime
import glob
import theano.sandbox


# Load the model (i.o. Model/model.json)
def load_model(json_path):
    model = model_from_json(open(json_path).read())
    return model


# Load the weights for model (i.o. Model/weights_L1L2.mat)
def load_weights(model, weight_path):
    dict2 = loadmat(weight_path)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        layer.set_weights(dict[str(i)])
        i += 1
    return model


# For load_weights
def conv_dict(dict2):
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict


# save model and weights
def save_model(model, json_path, weight_path):
    json_string = model.to_json()
    open(json_path, 'w').write(json_string)

    dict = {}
    i = 0
    for layer in model.layers:
        weights = layer.get_weights()
        weights_list = np.zeros(len(weights), dtype=np.object)
        weights_list[:] = weights
        dict[str(i)] = weights_list
        i += 1
    savemat(weight_path, dict)


# To ignore hidden files
def listdir_nohidden(AllVideos_Path):
    file_dir_extension = os.path.join(AllVideos_Path, '*_C.txt')
    for f in glob.glob(file_dir_extension):
        if not f.startswith('.'):
            yield os.path.basename(f)


def load_dataset_One_Video_Features(Test_Video_Path):
    f = open(Test_Video_Path, "r")
    feat_row = f.read().split('\n')
    VideoFeatue = []

    for feat_count in range(0, len(feat_row)):
        feat = np.float32(feat_row[feat_count].split()[:])
        if feat_count == 0:
            VideoFeatue = feat
        else:
            VideoFeatue = np.vstack((VideoFeatue, feat))

    return VideoFeatue


def load_video_feature(Videos_Path, list_iter):
    All_Videos = sorted(listdir_nohidden(Videos_Path))
    All_Videos.sort()

    All_VideoFeatues = []
    Video_count = 0
    for iv in list_iter:
        VideoPath = os.path.join(Videos_Path, All_Videos[iv])
        VideoFeatue = load_dataset_One_Video_Features(VideoPath)

        if Video_count == 0:
            All_VideoFeatues = VideoFeatue
        else:
            All_VideoFeatues = np.vstack((All_VideoFeatues, VideoFeatue))

        Video_count += 1

    return All_VideoFeatues


# We assume that the features of abnormal videos and normal videos are located in two different folders,
# respectively, AbnormalPath and AbnormalPath.
def load_dataset_Train_batch(AbnormalPath, NormalPath):
    # Each batch contains 60 videos.
    batchsize = 60
    # Haif of batch are abnormal videos
    n_exp = batchsize / 2
    # Number of abnormal videos for training
    Num_abnormal = 810
    # Number of Normal videos for training
    Num_Normal = 800

    # from Num_abnormal index randomly select the last n_exp indexes for abnormal videos
    Abnor_list_iter = np.random.permutation(Num_abnormal)
    Abnor_list_iter = Abnor_list_iter[Num_abnormal - n_exp:]
    # from Num_Normal index randomly select the last n_exp indexes for normal videos
    Norm_list_iter = np.random.permutation(Num_Normal)
    Norm_list_iter = Norm_list_iter[Num_Normal - n_exp:]

    # load video features of abnormal videos of a batch
    print("start to load abnormal video features")
    Abnor_video_features = load_video_feature(AbnormalPath, Abnor_list_iter)
    print("abnormal video features have been loaded")

    # load video features of normal videos of a batch
    print("start to load normal video features")
    Norm_video_features = load_video_feature(NormalPath, Norm_list_iter)
    print("normal video features have been loaded")

    # concatenate video features of abnormal and normal
    AllFeatures = np.vstack((Abnor_video_features, Norm_video_features))

    AllLabels = np.zeros(32 * batchsize, dtype='uint8')

    for iv in range(0, 32 * batchsize):
        # All instances of abnormal videos are labeled 0.
        # All instances of Normal videos are labeled 1.
        # This will be used in custom_objective to keep track of normal and abnormal videos indexes.
        if iv < n_exp * 32:
            AllLabels[iv] = int(0)
        else:
            AllLabels[iv] = int(1)

    return AllFeatures, AllLabels


def custom_objective(y_true, y_pred):
    y_true = T.flatten(y_true)
    y_pred = T.flatten(y_pred)

    # 32 segments per video
    n_seg = 32
    nvid = 60
    n_exp = int(nvid / 2)
    Num_d = 32 * nvid

    # sub_max represents the instants with highest score in bags (videos).
    sub_max = T.ones_like(y_pred)
    # sub_sum_labels is used to sum the labels in order to distinguish between normal and abnormal videos.
    sub_sum_labels = T.ones_like(y_true)
    # For holding the concatenation of summation of scores in the bag.
    sub_sum_l1 = T.ones_like(y_true)
    # For holding the concatenation of L2 of score in the bag.
    sub_l2 = T.ones_like(y_true)

    for ii in range(0, nvid, 1):
        # For Labels
        mm = y_true[ii * n_seg:(ii + 1) * n_seg]
        # Just to keep track of abnormal and normal vidoes
        sub_sum_labels = T.concatenate([sub_sum_labels, T.stack(T.sum(mm))])

        # For Feature scores
        Feat_Score = y_pred[ii * n_seg:(ii + 1) * n_seg]
        # Keep the maximum score of all instances in a Bag (video)
        sub_max = T.concatenate([sub_max, T.stack(T.max(Feat_Score))])
        # Keep the sum of scores of all instances in a Bag (video)
        sub_sum_l1 = T.concatenate([sub_sum_l1, T.stack(T.sum(Feat_Score))])

        z1 = T.ones_like(Feat_Score)
        z2 = T.concatenate([z1, Feat_Score])
        z3 = T.concatenate([Feat_Score, z1])
        z_22 = z2[31:]
        z_44 = z3[:33]
        z = z_22 - z_44
        z = z[1:32]
        z = T.sum(T.sqr(z))
        sub_l2 = T.concatenate([sub_l2, T.stack(z)])

    # We need this step since we have used T.ones_like
    sub_score = sub_max[Num_d:]
    # F_labels contains integer 32 for normal video and 0 for abnormal videos.
    F_labels = sub_sum_labels[Num_d:]

    # We need steps below since we have used T.ones_like
    sub_sum_l1 = sub_sum_l1[Num_d:]
    sub_sum_l1 = sub_sum_l1[:n_exp]
    sub_l2 = sub_l2[Num_d:]
    sub_l2 = sub_l2[:n_exp]

    # Index of normal videos
    # Since we labeled 1 for each of 32 segments of normal videos
    # F_labels=32 for normal video
    indx_nor = theano.tensor.eq(F_labels, 32).nonzero()[0]
    indx_abn = theano.tensor.eq(F_labels, 0).nonzero()[0]

    n_Nor = n_exp

    # maximum score of segment for each abnormal video
    Sub_Nor = sub_score[indx_nor]
    # maximum score of segment for each normal video
    Sub_Abn = sub_score[indx_abn]

    z = T.ones_like(y_true)
    for ii in range(0, n_Nor, 1):
        sub_z = T.maximum(1 - Sub_Abn + Sub_Nor[ii], 0)
        z = T.concatenate([z, T.stack(T.sum(sub_z))])

    z = z[Num_d:]
    # Final Loss
    z = T.mean(z, axis=-1) + 0.00008 * T.sum(sub_sum_l1) + 0.00008 * T.sum(sub_l2)

    return z


if __name__ == '__main__':

    print("Creating Model")
    model = Sequential()
    model.add(Dense(512, input_dim=4096, init='glorot_normal', W_regularizer=l2(0.001), activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(32, init='glorot_normal', W_regularizer=l2(0.001)))
    model.add(Dropout(0.6))
    model.add(Dense(1, init='glorot_normal', W_regularizer=l2(0.001), activation='sigmoid'))

    adagrad = Adagrad(lr=0.01, epsilon=1e-08)
    model.compile(loss=custom_objective, optimizer=adagrad)

    print("Starting training...")

    # AllClassPath contains C3D features of each video (.txt file). Each text file contains 32 features.
    AllClassPath = 'Train_Video_Feature/'

    # Output_dir is the directory where you want to save trained weights.
    output_dir = 'Model/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_path = output_dir + 'model.json'
    # weights.mat are the model weights that you will get after (or during) training.
    weights_path = output_dir + 'weights.mat'

    All_class_files = listdir(AllClassPath)
    All_class_files.sort()
    loss_graph = []
    num_iters = 20000
    time_before = datetime.now()

    for it_num in range(num_iters):
        # Path of abnormal already computed C3D features
        AbnormalPath = os.path.join(AllClassPath, All_class_files[0])
        # Path of Normal already computed C3D features
        NormalPath = os.path.join(AllClassPath, All_class_files[1])
        # Load normal and abnormal video C3D features
        inputs, targets = load_dataset_Train_batch(AbnormalPath, NormalPath)
        batch_loss = model.train_on_batch(inputs, targets)
        loss_graph = np.hstack((loss_graph, batch_loss))
        if it_num % 20 == 0:
            print("Iteration=" + str(iter) +
                  ", took " + str(datetime.now() - time_before) +
                  ", with loss of " + str(batch_loss))
            iteration_path = output_dir + 'Iterations_graph_' + str(iter) + '.mat'
            savemat(iteration_path, dict(loss_graph=loss_graph))
        # Save the model every 1000 iterations
        if it_num % 1000 == 0:
            weights_path = output_dir + 'weightsAnomalyL1L2_' + str(iter) + '.mat'
            save_model(model, model_path, weights_path)

    save_model(model, model_path, weights_path)
