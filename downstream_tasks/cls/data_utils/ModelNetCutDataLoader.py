import os
import numpy as np
import h5py
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    '''
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    return:
        centroids: sampled pointcloud index, [npoint, D]
    '''
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetCutDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

        # self.catfile = os.path.join(self.root, 'names.txt')
        # self.cat = [line.rstrip() for line in open(self.catfile)]
        # self.classes = dict(zip(self.cat, range(len(self.cat))))
        
        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'train_files.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'test_files.txt'))]

        assert (split == 'train' or split == 'test')
        self.datapath = shape_ids[split]

        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnetcut%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnetcut%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    f = h5py.File(fn, 'r')
                    cls = int(f['label'][:][0])
                    label = np.array([f['label'][:][0]]).astype(np.int32)
                    random_id = np.zeros((1,), dtype=int)
                    random_id = np.random.randint(30, size=1)
                    random_id = random_id + 1
                    if random_id[0] % 2 == 0:
                        cut1 = f['cut' + str(random_id[0]-1)][:]
                        cut2 = f['cut' + str(random_id[0])][:]
                    else:
                        cut1 = f['cut' + str(random_id[0])][:]
                        cut2 = f['cut' + str(random_id[0]+1)][:]
                    target = int(f['label'][:][0])
                    point_set = np.array(np.concatenate((cut1, cut2), axis=0)).astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            f = h5py.File(fn, 'r')
            cls = int(f['label'][:][0])
            label = np.array([f['label'][:][0]]).astype(np.int32)
            random_id = np.zeros((1,), dtype=int)
            random_id = np.random.randint(30, size=1)
            random_id = random_id + 1
            if random_id[0] % 2 == 0:
                cut1 = f['cut' + str(random_id[0]-1)][:]
                cut2 = f['cut' + str(random_id[0])][:]
            else:
                cut1 = f['cut' + str(random_id[0])][:]
                cut2 = f['cut' + str(random_id[0]+1)][:]
            target = int(f['label'][:][0])
            point_set = np.array(np.concatenate((cut1, cut2), axis=0)).astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)

if __name__ == '__main__':
    import torch

    data = ModelNetCutDataLoader('/data/modelnet40_ply_hdf5_2048_cut/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)




