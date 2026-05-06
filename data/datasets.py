"""Dataset classes used by CD-STeP training scripts.

The eight classes below cover three datasets:

    ACDC (cine MRI):
        - SemiSegDataset_VT       : training set, paired (sequence, single-frame label)
        - SemiSegDataset_VT_Int   : same but with explicit frame-index sampling
                                    (used by the frame-interval ablation)
        - SemiSegValidDataset     : validation set

    CAMUS (echocardiography):
        - SemiDatasetEcho_VT0     : training set
        - SemiDatasetEcho_VT0_Int : frame-interval ablation training set
        - SemiDatasetEchoValid    : validation set

    EchoNet-Dynamic (supplementary):
        - SemiDatasetEchoSeq      : half-sequence training (ED -> ES)
        - SemiDatasetEchoSeqValid : validation
"""
import csv
from abc import ABC
import cv2
import numpy as np
import torch
from torch.utils import data
import nibabel as nib
from scipy import ndimage
from numpy.linalg import inv
import torch.nn.functional as F
from collections import defaultdict
import random
import pandas as pd

# ---------------------------------------------------------------------------
# SemiSegDataset_VT
# ---------------------------------------------------------------------------
class SemiSegDataset_VT(data.Dataset, ABC):
    def __init__(self, data_dir, train_data_csv, train_data_gt, image_size=(224, 224, 18),
                 frame_number=20, mode='train',label_index=0,rand_frame=True):
        super(SemiSegDataset_VT, self).__init__()

        self.data_dir = data_dir
        self.train_data_csv = train_data_csv
        self.image_size = image_size
        self.mode = mode
        self.train_data_list = self.file2list(train_data_csv)
        self.train_data_gt_list = self.file2list(train_data_gt)
        self.rand_frame=rand_frame

        self.frame_number = frame_number
        self.label_index=label_index
        self.train_data = []
        self.train_data_gt = []

        self.frame_volumes = []

        self.slices = []
        self.slices_gt = []
        for index in range(len(self.train_data_list)):
            whole_img = self.getperdata(self.data_dir, self.train_data_list, index)
            image = self.data_preprocessing(whole_img, image_size=self.image_size) #H,W,N

            temp_image = np.expand_dims(image[:, :, :], axis=0)

            self.train_data.append(temp_image) # KT,H,W,N
        # idx_shape=(0,0,0)
        # for index in range(len(self.train_data_list)):
            
        #     whole_img = self.getperdata(self.data_dir, self.train_data_list, index,idx_shape)
        #     image = self.data_preprocessing(whole_img, image_size=self.image_size) #H,W,N
        #     idx_shape=image.shape

        #     temp_image = np.expand_dims(image[:, :, :], axis=0)

        #     self.train_data.append(temp_image) # KT,H,W,N

        for index in range(len(self.train_data_gt_list)):
            whole_label = self.getperdata(self.data_dir, self.train_data_gt_list, index) # ori have no idx_shape
            label = self.label_preprocessing(whole_label, self.image_size)

            self.train_data_gt.append(label)#K,H,W,N
        for _i in range(len(self.train_data) // self.frame_number):
            volumes = []
            for _j in range(self.frame_number):
                volumes.append(self.train_data[_j + self.frame_number * _i])
            frame_volumes = np.concatenate(volumes, axis=0)
            self.frame_volumes.append(frame_volumes)
        for index in range(len(self.frame_volumes)):
            whole_image = self.frame_volumes[index]
            for slices in range(whole_image.shape[-1]):
                self.slices.append(whole_image[:, :, :, slices])
            whole_gt = self.train_data_gt[index]
            for slices in range(whole_gt.shape[-1]):
                temp_gt = np.expand_dims(whole_gt[:, :, slices], axis=0)
                self.slices_gt.append(temp_gt)
    def __len__(self):
        return len(self.slices)

    def __getitem__(self, item):

        frame_img = self.slices[item]
        frame_image = torch.Tensor(frame_img)

        frame_gt = self.slices_gt[item]
        frame_gt = torch.LongTensor(frame_gt).squeeze()
        
        T = self.frame_number
        tov_target = torch.zeros(T, dtype=torch.float32)

        if self.rand_frame:
            if random.random() < 0.5:
                available_indices = [i for i in range(T) if i != self.label_index]
                
                if len(available_indices) >= 2:
                    # b. 随机、不重复地选择两个索引
                    idx1, idx2 = random.sample(available_indices, 2)

                    frame_image[[idx1, idx2]] = frame_image[[idx2, idx1]]
                    # frame_gt[[idx1, idx2]] = frame_gt[[idx2, idx1]]

                    # d. 更新 tov_target，
                    tov_target[idx1] = 1.0
                    tov_target[idx2] = 1.0

        return frame_image, frame_gt,tov_target

    def file2list(self, file_csv):
        img_list = []

        with open(file_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for line in reader:
                img_list.append(line['image_filenames'])

        return img_list

    def data_preprocessing(self, img, image_size):
        clip_min = np.percentile(img, 1)
        clip_max = np.percentile(img, 99)
        image = np.clip(img, clip_min, clip_max)
        image = (image - image.min()) / float(image.max() - image.min())
        x, y, z = image.shape
        x_centre, y_centre, z_center = int(x / 2), int(y / 2), int(z / 2)
        image = self.crop_image(image, x_centre, y_centre, z_center, image_size, constant_values=0)

        return image

    def getperdata(self, data_dir, data_list, index):
        image_name = data_dir + '/' + data_list[index]

        nib_image = nib.load(image_name)

        whole_image = nib_image.get_fdata().squeeze()

        whole_image = whole_image.astype('float32')

        return whole_image
    # def getperdata(self, data_dir, data_list, index,idx_shape=None):
    #     image_name = data_dir + '/' + data_list[index]
    #     try:
    #         nib_image = nib.load(image_name)

    #         whole_image = nib_image.get_fdata().squeeze()
    #     except:
    #         whole_image=np.zeros(idx_shape)

    #     whole_image = whole_image.astype('float32')

    #     return whole_image

    def crop_image(self, image, cx, cy, cz, size, constant_values=0):
        """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
        X, Y, Z = image.shape[:3]
        rX = size[0] // 2
        rY = size[1] // 2
        rZ = size[2] // 2
        x1, x2 = cx - rX, cx + (size[0] - rX)
        y1, y2 = cy - rY, cy + (size[1] - rY)
        z1, z2 = cz - rZ, cz + (size[2] - rZ)
        x1_, x2_ = max(x1, 0), min(x2, X)
        y1_, y2_ = max(y1, 0), min(y2, Y)
        z1_, z2_ = max(z1, 0), min(z2, Z)
        # Crop the image
        crop = image[x1_: x2_, y1_: y2_]
        # Pad the image if the specified size is larger than the input image size
        if crop.ndim == 3:
            crop = np.pad(crop,
                          ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_),(0,0)),
                          'constant', constant_values=constant_values)
        elif crop.ndim == 4:
            crop = np.pad(crop,
                          ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_),(0,0), (0, 0)),
                          'constant', constant_values=constant_values)
        else:
            print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
            exit(0)
        return crop

    def label_preprocessing(self, label, image_size):
        x, y, z = label.shape
        x_centre, y_centre, z_center = int(x / 2), int(y / 2), int(z / 2)
        label = self.crop_image(label, x_centre, y_centre, z_center, image_size)
        return label


# ---------------------------------------------------------------------------
# SemiSegDataset_VT_Int
# ---------------------------------------------------------------------------
class SemiSegDataset_VT_Int(data.Dataset, ABC):
    def __init__(self, data_dir, train_data_csv, train_data_gt, image_size=(224, 224, 18),
                 frame_number=20, mode='train',label_index=0,rand_frame=True):
        super(SemiSegDataset_VT_Int, self).__init__()

        self.data_dir = data_dir
        self.train_data_csv = train_data_csv
        self.image_size = image_size
        self.mode = mode
        self.train_data_list = self.file2list(train_data_csv)
        self.train_data_gt_list = self.file2list(train_data_gt)
        self.rand_frame=rand_frame

        self.frame_number = frame_number
        self.label_index=label_index
        self.train_data = []
        self.train_data_gt = []

        self.frame_volumes = []

        self.slices = []
        self.slices_gt = []
        idx_shape=(0,0,0)
        for index in range(len(self.train_data_list)):
            
            whole_img = self.getperdata(self.data_dir, self.train_data_list, index,idx_shape)
            image = self.data_preprocessing(whole_img, image_size=self.image_size) #H,W,N
            idx_shape=image.shape

            temp_image = np.expand_dims(image[:, :, :], axis=0)

            self.train_data.append(temp_image) # KT,H,W,N

        for index in range(len(self.train_data_gt_list)):
            whole_label = self.getperdata(self.data_dir, self.train_data_gt_list, index,idx_shape)
            label = self.label_preprocessing(whole_label, self.image_size)

            self.train_data_gt.append(label)#K,H,W,N
        for _i in range(len(self.train_data) // self.frame_number):
            volumes = []
            for _j in range(self.frame_number):
                volumes.append(self.train_data[_j + self.frame_number * _i])
            frame_volumes = np.concatenate(volumes, axis=0)
            self.frame_volumes.append(frame_volumes)
        for index in range(len(self.frame_volumes)):
            whole_image = self.frame_volumes[index]
            for slices in range(whole_image.shape[-1]):
                self.slices.append(whole_image[:, :, :, slices])
            whole_gt = self.train_data_gt[index]
            for slices in range(whole_gt.shape[-1]):
                temp_gt = np.expand_dims(whole_gt[:, :, slices], axis=0)
                self.slices_gt.append(temp_gt)
    def __len__(self):
        return len(self.slices)

    def __getitem__(self, item):

        frame_img = self.slices[item]
        frame_image = torch.Tensor(frame_img)

        frame_gt = self.slices_gt[item]
        frame_gt = torch.LongTensor(frame_gt).squeeze()
        
        T = self.frame_number
        tov_target = torch.zeros(T, dtype=torch.float32)

        if self.rand_frame:
            if random.random() < 0.5:
                available_indices = [i for i in range(T) if i != self.label_index]
                
                if len(available_indices) >= 2:
                    # b. 随机、不重复地选择两个索引
                    idx1, idx2 = random.sample(available_indices, 2)

                    frame_image[[idx1, idx2]] = frame_image[[idx2, idx1]]
                    # frame_gt[[idx1, idx2]] = frame_gt[[idx2, idx1]]

                    # d. 更新 tov_target，
                    tov_target[idx1] = 1.0
                    tov_target[idx2] = 1.0

        return frame_image, frame_gt,tov_target

    def file2list(self, file_csv):
        img_list = []

        with open(file_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for line in reader:
                img_list.append(line['image_filenames'])

        return img_list

    def data_preprocessing(self, img, image_size):
        if img.max() !=img.min():
            clip_min = np.percentile(img, 1)
            clip_max = np.percentile(img, 99)
            image = np.clip(img, clip_min, clip_max)
        else:
            image=img
        image = (image - image.min()) / float(image.max() - image.min()+1e-8)
        x, y, z = image.shape
        x_centre, y_centre, z_center = int(x / 2), int(y / 2), int(z / 2)
        image = self.crop_image(image, x_centre, y_centre, z_center, image_size, constant_values=0)

        return image

    def getperdata(self, data_dir, data_list, index,idx_shape):
        image_name = data_dir + '/' + data_list[index]
        try:
            nib_image = nib.load(image_name)

            whole_image = nib_image.get_fdata().squeeze()
        except:
            whole_image=np.zeros(idx_shape)

        whole_image = whole_image.astype('float32')

        return whole_image

    def crop_image(self, image, cx, cy, cz, size, constant_values=0):
        """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
        X, Y, Z = image.shape[:3]
        rX = size[0] // 2
        rY = size[1] // 2
        rZ = size[2] // 2
        x1, x2 = cx - rX, cx + (size[0] - rX)
        y1, y2 = cy - rY, cy + (size[1] - rY)
        z1, z2 = cz - rZ, cz + (size[2] - rZ)
        x1_, x2_ = max(x1, 0), min(x2, X)
        y1_, y2_ = max(y1, 0), min(y2, Y)
        z1_, z2_ = max(z1, 0), min(z2, Z)
        # Crop the image
        crop = image[x1_: x2_, y1_: y2_]
        # Pad the image if the specified size is larger than the input image size
        if crop.ndim == 3:
            crop = np.pad(crop,
                          ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_),(0,0)),
                          'constant', constant_values=constant_values)
        elif crop.ndim == 4:
            crop = np.pad(crop,
                          ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_),(0,0), (0, 0)),
                          'constant', constant_values=constant_values)
        else:
            print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
            exit(0)
        return crop

    def label_preprocessing(self, label, image_size):
        x, y, z = label.shape
        x_centre, y_centre, z_center = int(x / 2), int(y / 2), int(z / 2)
        label = self.crop_image(label, x_centre, y_centre, z_center, image_size)
        return label


# ---------------------------------------------------------------------------
# SemiSegValidDataset
# ---------------------------------------------------------------------------
class SemiSegValidDataset(data.Dataset, ABC):
    def __init__(self,
                 data_dir,
                 valid_data_csv,
                 image_size=(224, 224),
                 ):
        self.data_dir = data_dir
        self.valid_data_csv = valid_data_csv
        self.image_size = image_size

        self.img_file_list, self.label_file_list = self.file2list(self.valid_data_csv)

        self.img_list = []
        self.label_list = []

        for index in range(len(self.img_file_list)):
            whole_img = self.getperdata(self.data_dir, self.img_file_list, index)
            whole_label = self.getperdata(self.data_dir, self.label_file_list, index)
            image = self.data_preprocessing(whole_img, image_size)
            label = self.label_preprocessing(whole_label, image_size)

            for i in range(image.shape[2]):
                temp_image = np.expand_dims(image[:, :, i], axis=0)
                self.img_list.append(temp_image)
            for j in range(label.shape[2]):
                temp_label = np.expand_dims(label[:, :, j], axis=0)
                self.label_list.append(temp_label)

    def __getitem__(self, item):
        img = self.img_list[item]
        label = self.label_list[item].squeeze()

        image = torch.Tensor(img)
        target = torch.LongTensor(label)

        return image, target

    def __len__(self):
        return len(self.img_list)

    def file2list(self, file_csv):
        img_list = []
        label_list = []

        with open(file_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for line in reader:
                img_list.append(line['image_filenames'])
                label_list.append(line['label_filenames'])

        return img_list, label_list

    def data_preprocessing(self, img, image_size):
        clip_min = np.percentile(img, 1)
        clip_max = np.percentile(img, 99)
        image = np.clip(img, clip_min, clip_max)
        image = (image - image.min()) / float(image.max() - image.min())
        x, y, z = image.shape
        x_centre, y_centre = int(x / 2), int(y / 2)
        image = self.crop_image(image, x_centre, y_centre, image_size, constant_values=0)

        return image

    def getperdata(self, data_dir, data_list, index):
        image_name = data_dir + '/' + data_list[index]

        nib_image = nib.load(image_name)

        whole_image = nib_image.get_fdata().squeeze()

        whole_image = whole_image.astype('float32')

        return whole_image

    def crop_image(self, image, cx, cy, size, constant_values=0):
        """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
        X, Y = image.shape[:2]
        rX = size[0] // 2
        rY = size[1] // 2
        x1, x2 = cx - rX, cx + (size[0] - rX)
        y1, y2 = cy - rY, cy + (size[1] - rY)
        x1_, x2_ = max(x1, 0), min(x2, X)
        y1_, y2_ = max(y1, 0), min(y2, Y)
        # Crop the image
        crop = image[x1_: x2_, y1_: y2_]
        # Pad the image if the specified size is larger than the input image size
        if crop.ndim == 3:
            crop = np.pad(crop,
                          ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)),
                          'constant', constant_values=constant_values)
        elif crop.ndim == 4:
            crop = np.pad(crop,
                          ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0), (0, 0)),
                          'constant', constant_values=constant_values)
        else:
            print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
            exit(0)
        return crop

    def label_preprocessing(self, label, image_size):
        x, y, z = label.shape
        x_centre, y_centre = int(x / 2), int(y / 2)
        label = self.crop_image(label, x_centre, y_centre, image_size)
        return label



# ---------------------------------------------------------------------------
# SemiDatasetEcho_VT0
# ---------------------------------------------------------------------------
class SemiDatasetEcho_VT0(data.Dataset, ABC):
    def __init__(self,
                 data_dir,
                 data_csv,
                 gt_csv,
                 image_size=(224, 224, 20),
                 mode='train',
                 frame_number=20,
                 label_index=0,
                 rand_frame=True
                 ):
        self.data_dir = data_dir
        self.data_csv = data_csv
        self.gt_csv = gt_csv
        self.image_size = image_size
        self.mode = mode
        self.frame_number = frame_number
        self.train_data_list = self.file2list(data_csv)
        self.train_gt_list = self.file2list(gt_csv)
        self.label_index = label_index
        self.rand_frame=rand_frame
        self.train_data = []
        self.train_gt = []
        assert self.frame_number == image_size[-1]
        self.frame_volumes = []
        for index in range(len(self.train_data_list)):
            whole_image = self.getperdata(data_dir, self.train_data_list, index)
            image = self.data_preprocessing(whole_image, image_size)
            for i in range(image.shape[-1]):
                temp_image = np.expand_dims(image[:, :, i], axis=0)
                self.train_data.append(temp_image)

            whole_label = self.getperdata(self.data_dir, self.train_gt_list, index)
            label = self.label_preprocessing(whole_label, self.image_size)
            print(label.shape)
            self.train_gt.append(label[:, :, self.label_index])


        for _i in range(len(self.train_data) // self.frame_number):
            volumes = []
            for _j in range(self.frame_number):
                volumes.append(self.train_data[_i * self.frame_number + _j])
            frame_volumes = np.concatenate(volumes, axis=0)
            self.frame_volumes.append(frame_volumes)

    def __len__(self):
        return len(self.frame_volumes)

    def __getitem__(self, item):
        frame_img = self.frame_volumes[item]
        frame_image = torch.Tensor(frame_img)

        frame_gt = self.train_gt[item]
        frame_gt = torch.LongTensor(frame_gt)

        T = self.frame_number
        tov_target = torch.zeros(T, dtype=torch.float32)

        if self.rand_frame:
            if random.random() < 0.5:
                available_indices = [i for i in range(T) if i != self.label_index]
                
                if len(available_indices) >= 2:
                    # b. 随机、不重复地选择两个索引
                    idx1, idx2 = random.sample(available_indices, 2)

                    frame_image[[idx1, idx2]] = frame_image[[idx2, idx1]]
                    # frame_gt[[idx1, idx2]] = frame_gt[[idx2, idx1]]

                    # d. 更新 tov_target，
                    tov_target[idx1] = 1.0
                    tov_target[idx2] = 1.0

        return frame_image, frame_gt,tov_target

    def file2list(self, file_csv):
        img_list = []

        with open(file_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for line in reader:
                img_list.append(line['image_filenames'])

        return img_list

    def label_preprocessing(self, label, image_size):
        x, y, z = label.shape
        x_centre, y_centre, z_center = int(x / 2), int(y / 2), int(z / 2)
        label = self.crop_image(label, x_centre, y_centre, z_center, image_size)
        return label

    def data_preprocessing(self, img, image_size):
        clip_min = np.percentile(img, 1)
        clip_max = np.percentile(img, 99)
        image = np.clip(img, clip_min, clip_max)
        image = (image - image.min()) / float(image.max() - image.min())
        x, y, z = image.shape
        x_centre, y_centre, z_center = int(x / 2), int(y / 2), int(z / 2)
        image = self.crop_image(image, x_centre, y_centre, z_center, image_size, constant_values=0)

        return image

    def getperdata(self, data_dir, data_list, index):
        image_name = data_dir + '/' + data_list[index]

        nib_image = nib.load(image_name)

        whole_image = nib_image.get_fdata().squeeze()

        whole_image = whole_image.astype('float32')

        return whole_image

    # def crop_image(self, image, cx, cy, cz, size, constant_values=0):
    #     """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
    #     X, Y, Z = image.shape[:3]
    #     rX = size[0] // 2
    #     rY = size[1] // 2
    #     rZ = size[2] // 2
    #     x1, x2 = cx - rX, cx + (size[0] - rX)
    #     y1, y2 = cy - rY, cy + (size[1] - rY)
    #     z1, z2 = cz - rZ, cz + (size[2] - rZ)
    #     x1_, x2_ = max(x1, 0), min(x2, X)
    #     y1_, y2_ = max(y1, 0), min(y2, Y)
    #     z1_, z2_ = max(z1, 0), min(z2, Z)
    #     # Crop the image
    #     crop = image[x1_: x2_, y1_: y2_, z1_: z2_]
    #     if crop.ndim == 3:
    #         crop = np.pad(crop,
    #                       ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (z1_ - z1, z2 - z2_)),
    #                       'constant', constant_values=constant_values)
    #     elif crop.ndim == 4:
    #         crop = np.pad(crop,
    #                       ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (z1_ - z1, z2 - z2_), (0, 0)),
    #                       'constant', constant_values=constant_values)
    #     else:
    #         print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
    #         exit(0)
    #     return crop
    def crop_image(self, image, cx, cy, cz, size, constant_values=0):
        if image.ndim < 3:
            raise ValueError(f"Input image must have at least 3 dimensions, but got {image.ndim}")

        X, Y, Z = image.shape[:3]

        rX = size[0] // 2
        rY = size[1] // 2
        x1, x2 = cx - rX, cx + (size[0] - rX)
        y1, y2 = cy - rY, cy + (size[1] - rY)

        x1_, x2_ = max(x1, 0), min(x2, X)
        y1_, y2_ = max(y1, 0), min(y2, Y)

        z1 = 0
        z2 = size[2]

        z1_, z2_ = max(z1, 0), min(z2, Z)

        crop = image[x1_:x2_, y1_:y2_, z1_:z2_]

        # --- Padding ---
        # Calculate the required padding for each dimension.
        # For spatial dimensions (X, Y), padding can be on both sides.
        pad_x_before = x1_ - x1
        pad_x_after = x2 - x2_
        pad_y_before = y1_ - y1
        pad_y_after = y2 - y2_
        pad_z_before = z1_ - z1  # This will be 0 if cz >= 0, handles edge case of negative cz
        pad_z_after = z2 - z2_   # This pads the future frames if the sequence is too short

        # Define the padding widths for np.pad
        # The structure is ((before_ax0, after_ax0), (before_ax1, after_ax1), ...)
        padding_widths = [
            (pad_x_before, pad_x_after),
            (pad_y_before, pad_y_after),
            (pad_z_before, pad_z_after)
        ]
        if image.ndim > 3:
            remaining_dims = image.ndim - 3
            padding_widths.extend([(0, 0)] * remaining_dims)
        crop_padded = np.pad(crop,
                             padding_widths,
                             'constant',
                             constant_values=constant_values)
        
        return crop_padded


# ---------------------------------------------------------------------------
# SemiDatasetEcho_VT0_Int
# ---------------------------------------------------------------------------
class SemiDatasetEcho_VT0_Int(data.Dataset, ABC):
    def __init__(self,
                 data_dir,
                 data_csv,
                 gt_csv,
                 image_size=(224, 224, 20),
                 mode='train',
                 frame_number=20,
                 label_index=0,
                 rand_frame=True,
                 interval=0
                 ):
        self.data_dir = data_dir
        self.data_csv = data_csv
        self.gt_csv = gt_csv
        self.image_size = image_size
        self.mode = mode
        self.frame_number = frame_number
        self.train_data_list = self.file2list(data_csv)
        self.train_gt_list = self.file2list(gt_csv)
        self.label_index = label_index
        self.rand_frame=rand_frame
        self.train_data = []
        self.train_gt = []
        self.interval=interval
        assert self.frame_number == image_size[-1]
        self.frame_volumes = []
        for index in range(len(self.train_data_list)):
            whole_image = self.getperdata(data_dir, self.train_data_list, index)
            image = self.data_preprocessing(whole_image, image_size,self.interval)
            for i in range(image.shape[-1]):
                temp_image = np.expand_dims(image[:, :, i], axis=0)
                self.train_data.append(temp_image)

            whole_label = self.getperdata(self.data_dir, self.train_gt_list, index)
            label = self.label_preprocessing(whole_label, self.image_size)

            self.train_gt.append(label[:, :, self.label_index])

        for _i in range(len(self.train_data) // self.frame_number):
            volumes = []
            for _j in range(self.frame_number):
                volumes.append(self.train_data[_i * self.frame_number + _j])
            frame_volumes = np.concatenate(volumes, axis=0)
            self.frame_volumes.append(frame_volumes)

    def __len__(self):
        return len(self.frame_volumes)

    def __getitem__(self, item):
        frame_img = self.frame_volumes[item]
        frame_image = torch.Tensor(frame_img)

        frame_gt = self.train_gt[item]
        frame_gt = torch.LongTensor(frame_gt)

        T = self.frame_number
        tov_target = torch.zeros(T, dtype=torch.float32)

        if self.rand_frame:
            if random.random() < 0.5:
                available_indices = [i for i in range(T) if i != self.label_index]
                
                if len(available_indices) >= 2:
                    idx1, idx2 = random.sample(available_indices, 2)

                    frame_image[[idx1, idx2]] = frame_image[[idx2, idx1]]
                    # frame_gt[[idx1, idx2]] = frame_gt[[idx2, idx1]]

                    tov_target[idx1] = 1.0
                    tov_target[idx2] = 1.0

        return frame_image, frame_gt,tov_target

    def file2list(self, file_csv):
        img_list = []

        with open(file_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for line in reader:
                img_list.append(line['image_filenames'])

        return img_list

    def label_preprocessing(self, label, image_size):
        x, y, z = label.shape
        x_centre, y_centre, z_center = int(x / 2), int(y / 2), int(z / 2)
        label = self.crop_image(label, x_centre, y_centre, z_center, image_size)
        return label

    def data_preprocessing(self, img, image_size,interval=0):
        clip_min = np.percentile(img, 1)
        clip_max = np.percentile(img, 99)
        if interval>0:
            T,N,K=img.shape[-1],image_size[-1],interval
            img=np.pad(img, ((0, 0), (0, 0), (0, max(0, (N - 1) * interval + 1 - T))), 'constant')[:, :, np.arange(N) * K]
        image = np.clip(img, clip_min, clip_max)
        image = (image - image.min()) / float(image.max() - image.min())
        x, y, z = image.shape
        x_centre, y_centre, z_center = int(x / 2), int(y / 2), int(z / 2)
        image = self.crop_image(image, x_centre, y_centre, z_center, image_size, constant_values=0)

        return image

    def getperdata(self, data_dir, data_list, index):
        image_name = data_dir + '/' + data_list[index]

        nib_image = nib.load(image_name)

        whole_image = nib_image.get_fdata().squeeze()

        whole_image = whole_image.astype('float32')

        return whole_image

    # def crop_image(self, image, cx, cy, cz, size, constant_values=0):
    #     """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
    #     X, Y, Z = image.shape[:3]
    #     rX = size[0] // 2
    #     rY = size[1] // 2
    #     rZ = size[2] // 2
    #     x1, x2 = cx - rX, cx + (size[0] - rX)
    #     y1, y2 = cy - rY, cy + (size[1] - rY)
    #     z1, z2 = cz - rZ, cz + (size[2] - rZ)
    #     x1_, x2_ = max(x1, 0), min(x2, X)
    #     y1_, y2_ = max(y1, 0), min(y2, Y)
    #     z1_, z2_ = max(z1, 0), min(z2, Z)
    #     # Crop the image
    #     crop = image[x1_: x2_, y1_: y2_, z1_: z2_]
    #     if crop.ndim == 3:
    #         crop = np.pad(crop,
    #                       ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (z1_ - z1, z2 - z2_)),
    #                       'constant', constant_values=constant_values)
    #     elif crop.ndim == 4:
    #         crop = np.pad(crop,
    #                       ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (z1_ - z1, z2 - z2_), (0, 0)),
    #                       'constant', constant_values=constant_values)
    #     else:
    #         print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
    #         exit(0)
    #     return crop
    def crop_image(self, image, cx, cy, cz, size, constant_values=0):
        if image.ndim < 3:
            raise ValueError(f"Input image must have at least 3 dimensions, but got {image.ndim}")

        X, Y, Z = image.shape[:3]

        rX = size[0] // 2
        rY = size[1] // 2
        x1, x2 = cx - rX, cx + (size[0] - rX)
        y1, y2 = cy - rY, cy + (size[1] - rY)

        x1_, x2_ = max(x1, 0), min(x2, X)
        y1_, y2_ = max(y1, 0), min(y2, Y)

        z1 = 0
        z2 = size[2]

        z1_, z2_ = max(z1, 0), min(z2, Z)

        crop = image[x1_:x2_, y1_:y2_, z1_:z2_]

        # --- Padding ---
        # Calculate the required padding for each dimension.
        # For spatial dimensions (X, Y), padding can be on both sides.
        pad_x_before = x1_ - x1
        pad_x_after = x2 - x2_
        pad_y_before = y1_ - y1
        pad_y_after = y2 - y2_
        pad_z_before = z1_ - z1  # This will be 0 if cz >= 0, handles edge case of negative cz
        pad_z_after = z2 - z2_   # This pads the future frames if the sequence is too short

        # Define the padding widths for np.pad
        # The structure is ((before_ax0, after_ax0), (before_ax1, after_ax1), ...)
        padding_widths = [
            (pad_x_before, pad_x_after),
            (pad_y_before, pad_y_after),
            (pad_z_before, pad_z_after)
        ]
        if image.ndim > 3:
            remaining_dims = image.ndim - 3
            padding_widths.extend([(0, 0)] * remaining_dims)
        crop_padded = np.pad(crop,
                             padding_widths,
                             'constant',
                             constant_values=constant_values)
        
        return crop_padded


# ---------------------------------------------------------------------------
# SemiDatasetEchoSeq
# ---------------------------------------------------------------------------
class SemiDatasetEchoSeq(data.Dataset, ABC):
    """
    Single-frame-supervised variable-length sub-sequence dataset
    for EchoNet-Dynamic.

    Each sample is a window of `frame_number` consecutive frames starting
    at an annotated frame (ED or ES). The first frame (label_index=0)
    carries the GT; the rest are unlabeled neighbours.

    Args
    ----
    data_dir      : root folder under which the CSV relative paths resolve
    data_csv      : csv of image relative paths  (column 'image_filenames')
    gt_csv        : csv of label relative paths  (column 'image_filenames')
    image_size    : (H, W, N). image_size[-1] must equal frame_number.
    frame_number  : N, the number of frames per training sample. Can differ
                    between experiments -- the dataset on disk doesn't
                    need to be rebuilt.
    label_index   : which slice carries the GT. For this dataset's output
                    layout the anchor is always at t=0, so this is 0.
                    Kept as a parameter only to match the original trainer
                    signature.
    rand_frame    : if True, with prob 0.5 swap two non-anchor frames and
                    mark them in tov_target (matches original behaviour).

    Samples with segment length < frame_number are silently dropped at
    init. A summary is printed so the user knows how many.
    """

    def __init__(self,
                 data_dir,
                 data_csv,
                 gt_csv,
                 image_size=(224, 224, 6),
                 mode="train",
                 frame_number=6,
                 label_index=0,
                 rand_frame=False):
        self.data_dir     = data_dir
        self.image_size   = image_size
        self.mode         = mode
        self.frame_number = frame_number
        self.label_index  = label_index
        self.rand_frame   = rand_frame
        assert self.frame_number == image_size[-1], (
            f"frame_number={frame_number} must match image_size[-1]={image_size[-1]}"
        )
        assert self.label_index == 0, (
            "SemiDatasetEchoSeq assumes the anchor sits at t=0 (label_index=0). "
            f"Got label_index={label_index}."
        )

        self.image_rel = self._file2list(data_csv)
        self.label_rel = self._file2list(gt_csv)
        assert len(self.image_rel) == len(self.label_rel), (
            f"image/label csv length mismatch: {len(self.image_rel)} vs "
            f"{len(self.label_rel)}"
        )

        # Preload everything into RAM (same policy as SemiDatasetEcho_VT0).
        # 10k segments * N frames * 112x112 uint8 -> ~750 MB for N=6,
        # ~2.5 GB for N=20, etc. Switch to lazy loading if this is too
        # large for your box.
        self.frame_volumes = []   # list of (N, H, W) float32
        self.frame_gts     = []   # list of (H, W) int64, anchor frame only

        n_dropped_too_short = 0
        dropped_examples = []
        native_lengths = []

        for i, (img_rel, lab_rel) in enumerate(zip(self.image_rel, self.label_rel)):
            whole_image = self._load_nii(data_dir, img_rel)      # (W, H, L) or (H, W, L)
            whole_label = self._load_nii(data_dir, lab_rel)
            L_img = whole_image.shape[-1]
            L_lab = whole_label.shape[-1]
            assert L_img == L_lab, (
                f"[row {i}] image and label have different lengths: "
                f"{L_img} vs {L_lab} (img={img_rel})"
            )
            native_lengths.append(L_img)

            if L_img < self.frame_number:
                n_dropped_too_short += 1
                if len(dropped_examples) < 3:
                    dropped_examples.append((img_rel, L_img))
                continue

            # Take the first frame_number slices; anchor is at t=0 by
            # construction of the export script.
            img = self._preprocess_image(whole_image[:, :, :self.frame_number],
                                         image_size)           # (H, W, N)
            lab = self._preprocess_label(whole_label[:, :, :self.frame_number],
                                         image_size)           # (H, W, N)

            # Stack to (N, H, W) so the trainer's aug_images[:, i:i+1, :, :]
            # pattern (time axis first) works without transposing.
            frames_nhw = np.stack(
                [img[:, :, t] for t in range(self.frame_number)], axis=0)
            self.frame_volumes.append(frames_nhw.astype(np.float32))
            self.frame_gts.append(lab[:, :, self.label_index].astype(np.int64))

        # ------------------------------------------------------------------
        # Diagnostic summary: very important for reproducibility. Without
        # this the user can silently run with fewer samples than the CSV
        # suggests.
        n_total = len(self.image_rel)
        n_kept  = len(self.frame_volumes)
        print(f"[SemiDatasetEchoSeq] {n_kept}/{n_total} segments kept "
              f"(frame_number={self.frame_number}). "
              f"Dropped {n_dropped_too_short} segments with length < "
              f"{self.frame_number}.")
        if native_lengths:
            arr = np.asarray(native_lengths)
            print(f"[SemiDatasetEchoSeq] native segment length: "
                  f"min={arr.min()} max={arr.max()} mean={arr.mean():.1f} "
                  f"median={int(np.median(arr))}")
        if dropped_examples:
            print(f"[SemiDatasetEchoSeq] examples of dropped (too-short) segments:")
            for rel, L in dropped_examples:
                print(f"    {rel}   L={L}")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.frame_volumes)

    def __getitem__(self, item):
        frame_image = torch.from_numpy(self.frame_volumes[item])    # (N, H, W)
        frame_gt    = torch.from_numpy(self.frame_gts[item]).long() # (H, W)

        T = self.frame_number
        tov_target = torch.zeros(T, dtype=torch.float32)

        if self.rand_frame and random.random() < 0.5:
            available = [i for i in range(T) if i != self.label_index]
            if len(available) >= 2:
                i1, i2 = random.sample(available, 2)
                frame_image[[i1, i2]] = frame_image[[i2, i1]]
                tov_target[i1] = 1.0
                tov_target[i2] = 1.0

        return frame_image, frame_gt, tov_target

    # ------------------------------------------------------------------
    @staticmethod
    def _file2list(path):
        out = []
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append(row["image_filenames"])
        return out

    @staticmethod
    def _load_nii(data_dir, rel_path):
        arr = nib.load(f"{data_dir}/{rel_path}").get_fdata().squeeze()
        return arr.astype("float32")

    # Intensity normalise (percentile-clip + min-max) + center crop/pad to
    # (H, W, N). Same recipe as SemiDatasetEcho_VT0 so the trainer's
    # crop_batch_data shift math stays valid.
    def _preprocess_image(self, img, image_size):
        clip_min = np.percentile(img, 1)
        clip_max = np.percentile(img, 99)
        img = np.clip(img, clip_min, clip_max)
        denom = float(img.max() - img.min())
        img = (img - img.min()) / denom if denom > 0 else img * 0.0
        x, y, _ = img.shape
        return self._crop_image(img, x // 2, y // 2, image_size, constant_values=0.0)

    def _preprocess_label(self, lab, image_size):
        x, y, _ = lab.shape
        return self._crop_image(lab, x // 2, y // 2, image_size, constant_values=0)

    @staticmethod
    def _crop_image(image, cx, cy, size, constant_values=0):
        """Center-crop/pad axes 0,1 to (size[0], size[1]); preserve axis 2 length."""
        if image.ndim < 3:
            raise ValueError(f"expected >=3 dims, got {image.ndim}")
        X, Y, Z = image.shape[:3]
        rX, rY = size[0] // 2, size[1] // 2
        x1, x2 = cx - rX, cx + (size[0] - rX)
        y1, y2 = cy - rY, cy + (size[1] - rY)
        x1_, x2_ = max(x1, 0), min(x2, X)
        y1_, y2_ = max(y1, 0), min(y2, Y)
        z1_, z2_ = 0, min(size[2], Z)
        crop = image[x1_:x2_, y1_:y2_, z1_:z2_]
        pad = [(x1_ - x1, x2 - x2_),
               (y1_ - y1, y2 - y2_),
               (0,       size[2] - (z2_ - z1_))]
        if image.ndim > 3:
            pad += [(0, 0)] * (image.ndim - 3)
        return np.pad(crop, pad, mode="constant", constant_values=constant_values)


# ---------------------------------------------------------------------------
# SemiDatasetEchoValid
# ---------------------------------------------------------------------------
class SemiDatasetEchoValid(data.Dataset, ABC):
    def __init__(self,
                 data_dir,
                 valid_data_csv,
                 image_size=(224, 224),
                 ):
        self.data_dir = data_dir
        self.valid_data_csv = valid_data_csv
        self.image_size = image_size

        self.img_file_list, self.label_file_list = self.file2list(self.valid_data_csv)

        self.img_list = []
        self.label_list = []

        for index in range(len(self.img_file_list)):
            whole_img = self.getperdata(self.data_dir, self.img_file_list, index)
            whole_label = self.getperdata(self.data_dir, self.label_file_list, index)
            image = self.data_preprocessing(whole_img, image_size)
            label = self.label_preprocessing(whole_label, image_size)

            for i in range(image.shape[2]):
                temp_image = np.expand_dims(image[:, :, i], axis=0)
                self.img_list.append(temp_image)
            for j in range(label.shape[2]):
                temp_label = np.expand_dims(label[:, :, j], axis=0)
                self.label_list.append(temp_label)

    def __getitem__(self, item):
        img = self.img_list[item]
        label = self.label_list[item].squeeze()

        image = torch.Tensor(img)
        target = torch.LongTensor(label)

        return image, target

    def __len__(self):
        return len(self.img_list)

    def file2list(self, file_csv):
        img_list = []
        label_list = []

        with open(file_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for line in reader:
                img_list.append(line['image_filenames'])
                label_list.append(line['label_filenames'])

        return img_list, label_list

    def data_preprocessing(self, img, image_size):
        clip_min = np.percentile(img, 1)
        clip_max = np.percentile(img, 99)
        image = np.clip(img, clip_min, clip_max)
        image = (image - image.min()) / float(image.max() - image.min() + 1e-5)
        x, y, z = image.shape
        x_centre, y_centre = int(x / 2), int(y / 2)
        image = self.crop_image(image, x_centre, y_centre, image_size, constant_values=0)

        return image

    def getperdata(self, data_dir, data_list, index):
        image_name = data_dir + '/' + data_list[index]

        nib_image = nib.load(image_name)

        whole_image = nib_image.get_fdata().squeeze()

        whole_image = whole_image.astype('float32')

        return whole_image

    def crop_image(self, image, cx, cy, size, constant_values=0):
        """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
        X, Y = image.shape[:2]
        rX = size[0] // 2
        rY = size[1] // 2
        x1, x2 = cx - rX, cx + (size[0] - rX)
        y1, y2 = cy - rY, cy + (size[1] - rY)
        x1_, x2_ = max(x1, 0), min(x2, X)
        y1_, y2_ = max(y1, 0), min(y2, Y)
        # Crop the image
        crop = image[x1_: x2_, y1_: y2_]
        # Pad the image if the specified size is larger than the input image size
        if crop.ndim == 3:
            crop = np.pad(crop,
                          ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)),
                          'constant', constant_values=constant_values)
        elif crop.ndim == 4:
            crop = np.pad(crop,
                          ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0), (0, 0)),
                          'constant', constant_values=constant_values)
        else:
            print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
            exit(0)
        return crop

    def label_preprocessing(self, label, image_size):
        x, y, z = label.shape
        x_centre, y_centre = int(x / 2), int(y / 2)
        label = self.crop_image(label, x_centre, y_centre, image_size)
        return label



# ---------------------------------------------------------------------------
# SemiDatasetEchoSeqValid
# ---------------------------------------------------------------------------
class SemiDatasetEchoSeqValid(data.Dataset, ABC):
    def __init__(self,
                 data_dir,
                 valid_data_csv,
                 image_size=(224, 224),
                 ):
        self.data_dir = data_dir
        self.valid_data_csv = valid_data_csv
        self.image_size = image_size

        self.img_file_list, self.label_file_list = self.file2list(self.valid_data_csv)

        self.img_list = []
        self.label_list = []

        for index in range(len(self.img_file_list)):
            whole_img = self.getperdata(self.data_dir, self.img_file_list, index)
            whole_label = self.getperdata(self.data_dir, self.label_file_list, index)
            image = self.data_preprocessing(whole_img, image_size)
            label = self.label_preprocessing(whole_label, image_size)

            for i in range(image.shape[2]):
                temp_image = np.expand_dims(image[:, :, i], axis=0)
                self.img_list.append(temp_image)
            for j in range(label.shape[2]):
                temp_label = np.expand_dims(label[:, :, j], axis=0)
                self.label_list.append(temp_label)

    def __getitem__(self, item):
        img = self.img_list[item]
        label = self.label_list[item].squeeze()

        image = torch.Tensor(img)
        target = torch.LongTensor(label)

        return image, target

    def __len__(self):
        return len(self.img_list)

    def file2list(self, file_csv):
        img_list = []
        label_list = []

        with open(file_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for line in reader:
                img_list.append(line['image_filenames'])
                label_list.append(line['label_filenames'])

        return img_list, label_list

    def data_preprocessing(self, img, image_size):
        clip_min = np.percentile(img, 1)
        clip_max = np.percentile(img, 99)
        image = np.clip(img, clip_min, clip_max)
        image = (image - image.min()) / float(image.max() - image.min() + 1e-5)
        x, y, z = image.shape
        x_centre, y_centre = int(x / 2), int(y / 2)
        image = self.crop_image(image, x_centre, y_centre, image_size, constant_values=0)

        return image

    def getperdata(self, data_dir, data_list, index):
        image_name = data_dir + '/' + data_list[index]
        nib_image = nib.load(image_name)
        
        whole_image = nib_image.get_fdata().squeeze()

        # FIX: If squeeze() removed the 3rd dimension (making it 2D), add it back
        # so that it stays (H, W, 1) to prevent unpacking errors later.
        if whole_image.ndim == 2:
            whole_image = np.expand_dims(whole_image, axis=2)

        whole_image = whole_image.astype('float32')

        return whole_image

    def crop_image(self, image, cx, cy, size, constant_values=0):
        """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
        X, Y = image.shape[:2]
        rX = size[0] // 2
        rY = size[1] // 2
        x1, x2 = cx - rX, cx + (size[0] - rX)
        y1, y2 = cy - rY, cy + (size[1] - rY)
        x1_, x2_ = max(x1, 0), min(x2, X)
        y1_, y2_ = max(y1, 0), min(y2, Y)
        # Crop the image
        crop = image[x1_: x2_, y1_: y2_]
        # Pad the image if the specified size is larger than the input image size
        if crop.ndim == 3:
            crop = np.pad(crop,
                          ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)),
                          'constant', constant_values=constant_values)
        elif crop.ndim == 4:
            crop = np.pad(crop,
                          ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0), (0, 0)),
                          'constant', constant_values=constant_values)
        else:
            print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
            exit(0)
        return crop

    def label_preprocessing(self, label, image_size):
        x, y, z = label.shape
        x_centre, y_centre = int(x / 2), int(y / 2)
        label = self.crop_image(label, x_centre, y_centre, image_size)
        return label

