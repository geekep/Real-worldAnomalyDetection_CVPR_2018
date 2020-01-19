from scipy.io import savemat
import os
from os import listdir
import numpy
from datetime import datetime
import theano.sandbox
import TrainingAnomalyDetector_public as train

# theano.sandbox.cuda.use('gpu0')


seed = 7
numpy.random.seed(seed)

print("Starting testing...")

# AllTest_Video_Path contains C3D features (.txt file) of each video. Each text file contains 32 features.
AllTest_Video_Path = 'Testing_Video_Feature/'
All_Test_files = listdir(AllTest_Video_Path)
All_Test_files.sort()
nVideos = len(All_Test_files)
# Results_Path is the folder where you can save your test results.
Results_Path = 'Model_Res/'
if not os.path.exists(Results_Path):
    os.makedirs(Results_Path)
# Model_dir is the folder where we have placed our trained model and weights.
Model_dir = 'Model/'
# weights_path is Trained model weights.
weights_path = Model_dir + 'weights_L1L2.mat'
model_path = Model_dir + 'model.json'

model = train.load_model(model_path)
train.load_weights(model, weights_path)
time_before = datetime.now()

for iv in range(nVideos):
    Test_Video_Path = os.path.join(AllTest_Video_Path, All_Test_files[iv])
    # 32 segment features for one testing video
    inputs = train.load_dataset_One_Video_Features(Test_Video_Path)
    # Get anomaly prediction for each of 32 video segments.
    predictions = model.predict_on_batch(inputs)
    predictions_path = Results_Path + All_Test_files[iv][1:-4] + '.mat'
    # Save array of 1 * 32, containing anomaly score for each segment.
    savemat(predictions_path, {'prediction': predictions})
    print("Total Time took: " + str(datetime.now() - time_before))
