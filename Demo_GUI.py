from math import factorial
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PyQt5 import QtWidgets

import TrainingAnomalyDetector_public as train
# from Test_Anomaly_Detector_public import model

seed = 7
np.random.seed(seed)


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


class PrettyWidget(QtWidgets.QWidget):
    def __init__(self):
        super(PrettyWidget, self).__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(500, 100, 500, 500)
        self.setWindowTitle('Anomaly Detection')
        btn = QtWidgets.QPushButton('Anomaly Detection System\nPlease select a video', self)

        Model_dir = 'Model/'
        model_path = Model_dir + 'model.json'
        weights_path = Model_dir + 'weights_L1L2.mat'
        # load abnormality model
        global model
        model = train.load_model(model_path)
        train.load_weights(model, weights_path)

        btn.resize(btn.sizeHint())
        btn.clicked.connect(self.SingleBrowse)
        btn.move(150, 200)
        self.show()

    def SingleBrowse(self):
        video_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', "Testing_Videos")
        print(video_path[0])

        FeaturePath = 'Testing_C3D_Feature' + '/' + video_path[0].split('/')[0:-4]
        FeaturePath = FeaturePath + '.txt'
        inputs = train.load_dataset_One_Video_Features(FeaturePath)
        predictions = model.predict_on_batch(inputs)

        cap = cv2.VideoCapture(video_path[0])
        while not cap.isOpened():
            cap = cv2.VideoCapture(video_path)
            cv2.waitKey(1000)
            print("Wait for the header")
        Total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        total_segments = np.linspace(1, Total_frames, num=33)
        total_segments = total_segments.round()

        Frames_Score = []
        count = 0
        for iv in range(0, 32):
            F_Score = np.matlib.repmat(predictions[iv], 1, (int(total_segments[iv + 1]) - int(total_segments[iv])))
            if count == 0:
                Frames_Score = F_Score
            if count > 0:
                Frames_Score = np.hstack((Frames_Score, F_Score))

            count += 1

        print("Anomaly Prediction")
        x = np.linspace(1, Total_frames, Total_frames)
        scores = Frames_Score
        scores1 = scores.reshape((scores.shape[1],))
        scores1 = savitzky_golay(scores1, 101, 3)
        plt.close()
        break_pt = min(scores1.shape[0], x.shape[0])
        plt.axis([0, Total_frames, 0, 1])
        i = 0
        while True:
            flag, frame = cap.read()
            if flag:
                i = i + 1
                cv2.imshow('video', frame)
                jj = i % 25
                if jj == 1:
                    plt.plot(x[:i], scores1[:i], color='r', linewidth=3)
                    plt.draw()
                    plt.pause(0.000000000000000000000001)

                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                print(str(pos_frame) + " frames")
            else:
                # The next frame is not ready, so we try to read it again.
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
                print("frame is not ready")
                # It is better to wait for a while for the next frame to be ready
                cv2.waitKey(1000)

            if cv2.waitKey(10) == 27:
                break
            # Stop if the number of captured frames is equal to the total number of frames.
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == break_pt:
                break


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = PrettyWidget()
    app.exec_()
