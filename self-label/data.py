import os
import sys

import h5py
import numpy as np
import pickle
import torch
from tqdm import tqdm

import models


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
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


class ModelNetDataLoader(torch.utils.data.Dataset):

    def __init__(self,
                 root,
                 npoints=2048,
                 split='train',
                 evaluation=False,
                 supervised=False
                 ):
        self.npoints = npoints
        self.root = root
        self.evaluation = evaluation
        self.supervised = supervised
        self.num_category = 40
        self.process_data = True
        self.uniform = True
        self.use_normals = False

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
            shape_ids['trainval'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_trainval.txt'))]


        assert (split == 'train' or split == 'test' or split == 'trainval')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

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
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]
        
        if self.supervised == True:
            
            return point_set, label[0], index

        return point_set, 0, index

    def __getitem__(self, index):
        return self._get_item(index)
    

class ScannetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 npoints=1024,
                 config=[],
                 evaluation=False,
                 supervised=False
                 ):
        self.npoints = npoints
        self.root = root
        self.config = config
        self.evaluation = evaluation
        self.supervised = supervised
        name = self.config[0]
        save_path = self.config[1]
        save_path_label = self.config[2]
        arch = self.config[3]
        pseudo_labels_path = self.config[4]
        outs = self.config[5]

        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(name)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.list_of_points = [None] * len(self.fns)
        self.list_of_labels = [None] * len(self.fns)

        if not os.path.exists(save_path):
            for index in tqdm(range(len(self.fns)), total=len(self.fns)):
                fn = self.fns[index]
                f = h5py.File(fn, 'r')
                data = np.array(f['xyz']).astype(np.float32)

                data[:, 0:3] = pc_normalize(data[:, 0:3])
                data = farthest_point_sample(data, self.npoints)
                
                self.list_of_points[index] = data

            with open(save_path, 'wb') as f:
                pickle.dump([self.list_of_points], f)
        else:
            print('Load processed data from %s...' % save_path)
            with open(save_path, 'rb') as f:
                [self.list_of_points] = pickle.load(f)

        if self.evaluation == True or self.supervised == True:
            nyu40ids = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
            nyu40id2class = {nyu40id: i for i,nyu40id in enumerate(list(nyu40ids))}

            if not os.path.exists(save_path_label):
                print("label does not exist, creating one ...")
                if name == 'train' or name == 'val':
                    for index in tqdm(range(len(self.fns)), total=len(self.fns)):
                        fn = self.fns[index]
                        f = h5py.File(fn, 'r')
                        cls = int(f['prop'][6])
                        label = nyu40id2class[cls]
                        self.list_of_labels[index] = label
                else:
                    model = models.__dict__[arch](num_classes=outs, supervised=False, evaluation=False)
                    ckpt = torch.load(pseudo_labels_path)
                    model.load_state_dict(ckpt['state_dict'], strict=False)
                    labels = ckpt['L'][0]
                    labels = labels.cpu()
                    self.list_of_labels = labels.numpy()

                with open(save_path_label, 'wb') as f:
                    pickle.dump([self.list_of_labels], f)

            else:
                print('Load processed label data from %s...' % save_path_label)
                with open(save_path_label, 'rb') as f:
                    [self.list_of_labels] = pickle.load(f)

        
    def __getitem__(self, index):
        
        point_set = self.list_of_points[index]
        point_set = torch.from_numpy(point_set.astype(np.float32))

        if self.supervised == True:
            label = self.list_of_labels[index]
            return point_set, label, index

        return point_set, 0,index

    def __len__(self):
        return len(self.fns)


def get_dataloader(dataset, point_dir, batch_size=256,num_workers=8, split='train',
                   evaluation=False, supervised=False, npoints=1024, arch='', pseudo_labels_path='', outs=[]):
    name = '/'
    save_path = '/'
    save_path_label = '/'
    root = point_dir

    if dataset == 'scannet':
        if split == 'train':
                name = "train_files_gt_h5"
                save_path = os.path.join(root, 'scannet_object_gt_{npoints}pts.dat'.format(npoints=npoints))
                save_path_label = os.path.join(root, 'scannet_object_gt_labels_{npoints}pts.dat'.format(npoints=npoints))
                
        elif split == 'val':
            name = "test_files_gt_h5"
            save_path = os.path.join(root, 'scannet_object_gt_{npoints}pts_val.dat'.format(npoints=npoints))
            save_path_label = os.path.join(root, 'scannet_object_gt_labels_{npoints}pts_val.dat'.format(npoints=npoints))

        elif split == 'trainval':
            name = "trainval_files_gt_h5"
            save_path = os.path.join(root, 'scannet_object_gt_{npoints}pts_trainval.dat'.format(npoints=npoints))
            save_path_label = os.path.join(root, 'scannet_object_gt_labels_{npoints}pts_trainval.dat'.format(npoints=npoints))

        elif split == 'scannetv2_trainval':
            name = "trainval_files_gt_h5_trainval"
            save_path = os.path.join(root, 'scannetv2_object_gt_{npoints}pts_trainval.dat'.format(npoints=npoints))
            save_path_label = os.path.join(root, 'scannetv2_object_gt_labels_{npoints}pts_trainval.dat'.format(npoints=npoints))

        elif split == 'train_unsup':
            ckpt = pseudo_labels_path.split('/')
            ckpt = ckpt[1]
            name = "train_files_unsup_h5"
            save_path = os.path.join(root, 'scannet_object_unsup_{npoints}pts.dat'.format(npoints=npoints))
            save_path_label = os.path.join(root, 'scannet_object_unsup_labels_{npoints}pts_{ckpt}.dat'.format(npoints=npoints, ckpt=ckpt))

        elif split == 'val_unsup':
            name = "test_files_unsup_h5"
            save_path = os.path.join(root, 'scannet_object_unsup_{npoints}pts_val.dat'.format(npoints=npoints))
            save_path_label = os.path.join(root, 'scannet_object_unsup_labels_{npoints}pts_val.dat'.format(npoints=npoints))

        elif split == 'train_unsup_wypr':
            name = "train_files_unsup_h5_wypr"
            save_path = os.path.join(root, 'scannet_object_unsup_{npoints}pts_wypr.dat'.format(npoints=npoints))

        elif split == 'val_unsup_wypr':
            name = "test_files_unsup_h5_wypr"
            save_path = os.path.join(root, 'scannet_object_unsup_{npoints}pts_val_wypr.dat'.format(npoints=npoints))

        elif split == 'train_unsup_wypr_ensambled':
            name = "train_files_unsup_h5_wypr_ensambled"
            save_path = os.path.join(root, 'scannet_object_unsup_{npoints}pts_wypr_ensambled.dat'.format(npoints=npoints))

        elif split == 'val_unsup_wypr_ensambled':
            name = "test_files_unsup_h5_wypr_ensambled"
            save_path = os.path.join(root, 'scannet_object_unsup_{npoints}pts_val_wypr_ensambled.dat'.format(npoints=npoints))
    
        else:
            print("Unsupported dataset, please check the configuration again ...")
            sys.exit(1)

        config = [name, save_path, save_path_label, arch, pseudo_labels_path, outs]
        
        dataset = ScannetDataset(root=point_dir, config=config, evaluation=evaluation, supervised=supervised)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            # drop_last=True,
        )

    elif dataset == 'modelnet40':
        
        dataset = ModelNetDataLoader(root=point_dir, split=split, evaluation=evaluation, supervised=supervised)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
        
    return loader


def return_model_loader(args, return_loader=True):
    outs = [args.ncl]*args.hc
    assert args.arch in ['pointnet2','pointTransformer', 'point_transformer']

    model = models.__dict__[args.arch](num_classes=outs, supervised=args.supervised, evaluation=args.evaluation)
        
    if not return_loader:
        return model
    
    else:
        if args.supervised == True and args.evaluation == True:
            print("Error encountered, please specify what type of model e.g. self-supervised, supervised, or eval ...")
            sys.exit(1)

        if args.supervised ==True or args.evaluation == True:
            if args.supervised == True:
                print("Experiment on supervised model ...")
            else:
                print("Experiment on evaluation model ...")
            train = args.split[0]
            val = args.split[1]
            train_loader = get_dataloader(
                                          dataset=args.dataset,
                                          point_dir=args.data_path, 
                                          batch_size=args.batch_size, 
                                          num_workers=args.workers,
                                          split=train, 
                                          supervised=args.supervised, 
                                          evaluation=args.evaluation, 
                                          npoints=args.npoints,
                                          arch=args.arch,
                                          pseudo_labels_path=args.pseudo_labels_path,
                                          outs=outs
                                         )

            test_loader = get_dataloader(
                                         dataset=args.dataset,
                                         point_dir=args.data_path, 
                                         batch_size=args.batch_size, 
                                         num_workers=args.workers,
                                         split=val, 
                                         supervised=args.supervised, 
                                         evaluation=args.evaluation, 
                                         npoints=args.npoints,
                                         arch=args.arch,
                                         pseudo_labels_path=args.pseudo_labels_path,
                                         outs=outs
                                        )

            return model, train_loader, test_loader
        
        else:
            print("Experiment on self-supervised model ...")
            
            train_loader = get_dataloader(
                                          dataset=args.dataset,
                                          point_dir=args.data_path, 
                                          batch_size=args.batch_size, 
                                          num_workers=args.workers,
                                          split=args.split[0], 
                                          evaluation=args.evaluation, 
                                          supervised=args.supervised,
                                          npoints=args.npoints,
                                          arch=args.arch
                                         )

            return model, train_loader
