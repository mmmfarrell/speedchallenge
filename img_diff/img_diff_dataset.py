from __future__ import print_function

import cv2
import numpy as np
import os
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

class ImgDiffDataset(Dataset):

    def __init__(self, data_dir, outputs_file, val_split=0.2):
        self.data_dir = os.path.expanduser(data_dir)
        self.data_files = sorted(os.listdir(self.data_dir))
        self.num_imgs = len(self.data_files)

        with open(os.path.expanduser(outputs_file), "r") as txt_file:
            self.outputs = txt_file.read().splitlines()
            self.outputs = [float(i) for i in self.outputs]

        assert(len(self.outputs) == self.num_imgs)
        print("Loaded data directory with {} images".format(self.num_imgs))

        self._test_train_split(val_split)
        self.num_train = len(self.train_files)
        self.num_val = len(self.val_files)

    def _test_train_split(self, val_split):
        # For now, just grab the last chunk as validation
        num_val = int(val_split * self.num_imgs)
        num_train = self.num_imgs - num_val

        self.train_files = self.data_files[0:num_train]
        self.train_outputs = self.outputs[0:num_train]

        self.val_files = self.data_files[num_train:]
        self.val_outputs = self.outputs[num_train:]

    def __len__(self):
        return len(self.train_files) - 1

    def __getitem__(self, idx):

        # i = idx
        i = 1

        img0_file = os.path.join(self.data_dir, self.train_files[i - 1])
        img0 = cv2.imread(img0_file)

        img1_file = os.path.join(self.data_dir, self.train_files[i + 0])
        img1 = cv2.imread(img1_file)

        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        gray0 = gray0.astype(np.float32) / 255.
        gray1 = gray1.astype(np.float32) / 255.

        diff = np.abs(gray1 - gray0)
        # print("Speed: {}".format(self.train_outputs[i]))
        downsample = 8
        diff = cv2.resize(diff, (640 / downsample, 480 / downsample))
        diff = np.expand_dims(diff, axis=0)

        # cv2.imshow("0", gray0)
        # cv2.imshow("1", gray1)
        # cv2.imshow("diff", diff)
        # cv2.waitKey(1)
        speed = np.float32(self.train_outputs[i])
        # return diff, speed

        sample = {'image': diff, 'speed': speed}
        return sample



if __name__ == '__main__':
    data_dir = "~/jobs/comma/speedchallenge/data/train"
    outputs_file = "~/jobs/comma/speedchallenge/data/train.txt"
    dg = ImgDiffDataset(data_dir, outputs_file)
    # img, speed = dg.get_train_batch()
    # print(img)
    # print(speed)

    for i in range(len(dg)):
        sample = dg[i]

        print(i, sample['image'].shape, sample['speed'])

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(sample['image'])
        # show_landmarks(**sample)

        if i == 3:
            plt.show()
            break

    # img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("img", img_bgr)
    # cv2.waitKey(0)
