#!/usr/bin/env python
import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

#data processing step

#Downloads the dataset from source and unzip it
def download(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], data_dir))
        os.system('rm %s' % (zipfile))

#data preprocessing, convert each h5 files into two numpy arrays, one for data, the other for label
        #1.h5 files contains mutiple datasets
        #2.Access the data and label datasets
        #3.Put each into an array
def load_data(data_dir, partition):
    download(data_dir)
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        with h5py.File(h5_name, 'r') as f:#opens the file in read-only mode
            data = f['data'][:].astype('float32')#Access the dataset named "data" in the HDF5 file, and convert the data to a narray of float32
            label = f['label'][:].astype('int64')#Access the dataset named "label" in the HDF5 file, and convert the data to a narray of int64
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)#Each entry cotains to a point cloud corresponding to a single object
    all_label = np.concatenate(all_label, axis=0)#contains all labels within the edataset
    return all_data, all_label

"""
Point cloud is translated
"""
def translate_pointcloud(pointcloud):
    """
    for scaling and shifting the point cloud
    :param pointcloud:s
    :return:
    """
    scale = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    shift = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, scale), shift).astype('float32')
    return translated_pointcloud


class ModelNet40(Dataset):
    """
    This is the data loader for ModelNet 40
    ModelNet40 contains 12,311 meshed CAD models from 40 categories.

    num_points: 1024 by default
    data_dir
    paritition: train or test
    """
    def __init__(self, num_points=1024, data_dir="/data/deepgcn/modelnet40", partition='train'):
        self.data, self.label = load_data(data_dir, partition)
        self.num_points = num_points
        self.partition = partition

    """
    Retrieves a specific point cloud from the dataset and its label, and 
    apply optional transformation to it if it is part of the training set
    Implemented to make the class iterable
    """
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]#Access the point cloud at the specified index up to self.num_points number of points
        label = self.label[item]#retrives the label corresponding to the current point cloud from self.label narray
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)#apply transformation to pointcloud
            np.random.shuffle(pointcloud)
        return pointcloud, label

    """Implemented to make the class iterable"""    
    def __len__(self):
        return self.data.shape[0]

    def num_classes(self):
        return np.max(self.label) + 1

"""
Executed only if data.py is run directly
"""
if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    """
    Retrieves every point cloud and its corresponding labels, each point cloud have 1024 points. Print their dimensions.  
    Possible to convert output into xyz format and visulize the point cloud?
    """
    for data, label in train:
        print(data.shape)
        print(label.shape)
