import os
import sys
import numpy as np
import cv2
import torch
from torch.autograd import Variable
from C3D.C3D_model import C3D

parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)


def get_video_clip(cuboid):
    patchList = []
    for i in range(cuboid.shape[0]):
        patch = cuboid[i, :, :]
        patch = patch.astype('uint8')
        patch = cv2.resize(patch, (112, 112), interpolation=cv2.INTER_CUBIC)
        patchList.append(patch)

    cuboid = np.array(patchList)
    if cuboid.ndim < 4:
        cuboid = np.expand_dims(cuboid, axis=cuboid.ndim)
        cuboid = np.concatenate((cuboid, cuboid, cuboid), axis=-1)

    cuboid = cuboid.transpose(3, 0, 1, 2)  # channel, frameNum, height, width
    cuboid = np.expand_dims(cuboid, axis=0)  # batch axis
    cuboid = np.float32(cuboid)

    return torch.from_numpy(clip)


def main():

    net = C3D()
    preTrainedWeights = "./C3D/c3d.pickle"
    if os.path.exists(preTrainedWeights):
        net.load_state_dict(torch.load(preTrainedWeights))
    else:
        print("cannot load weights of pre-trained network model")

    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    X = torch.from_numpy()
    X = torch.from_numpy(cuboid)
    X = Variable(X)
    if torch.cuda.is_available():
        X = X.cuda()
    C3Dfeature = net(X)
    C3Dfeature = C3Dfeature.data.cpu().numpy()

    return C3Dfeature


if __name__ == '__main__':
    main()
