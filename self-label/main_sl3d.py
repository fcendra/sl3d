import argparse
import warnings
import os
import time

import numpy as np
import glob
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
from torchvision import transforms

from pointnet2_ops import pointnet2_utils
import provider
import files
import util
import sinkhornknopp as sk
from data import return_model_loader

try:
    from tensorboardX import SummaryWriter
except:
    pass

torch.cuda.empty_cache()
warnings.simplefilter("ignore", UserWarning)


class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2./3., scale_high=3. /2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range
    
    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])

            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()
        
        return pc

train_transforms = transforms.Compose(
    [
        PointcloudScaleAndTranslate(),
    ]
)


class Optimizer:
    def __init__(self, m, hc, ncl, t_loader, n_epochs, lr, weight_decay=1e-5, ckpt_dir='/', pretrained_model='', arch='', npoints=0):
        self.num_epochs = n_epochs
        self.lr = lr
        self.lr_schedule = lambda epoch: ((epoch < 350) * (self.lr * (0.1 ** (epoch // args.lrdrop)))
                                          + (epoch >= 350) * self.lr * 0.1 ** 3)
        self.momentum = 0.9
        self.weight_decay = weight_decay
        self.checkpoint_dir = ckpt_dir

        if list(glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint*.pth'))):
            self.resume = True
        else: 
            self.resume = False

        self.pretrained_model = pretrained_model
        self.writer = None
        self.arch = arch
        self.npoints = npoints

        # model stuff
        self.hc = hc
        self.K = ncl
        self.model = m
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nmodel_gpus = len(args.modeldevice)
        self.pseudo_loader = t_loader # can also be DataLoader with less aug.
        self.train_loader = t_loader
        self.lamb = args.lamb # the parameter lambda in the SK algorithm
        self.dtype = torch.float64 if not args.cpu else np.float64

        self.outs = [self.K]*args.hc
        # activations of previous to last layer to be saved if using multiple heads.
        self.presize = 256

    def optimize_labels(self, niter):
        if not args.cpu and torch.cuda.device_count() > 1:
            sk.gpu_sk(self)
        else:
            self.dtype = np.float64
            sk.cpu_sk(self)

        # save Label-assignments: optional
        torch.save(self.L, os.path.join(self.checkpoint_dir, 'L', str(niter) + '_L.gz'))

        # free memory
        self.PS = 0

    def optimize_epoch(self, optimizer, loader, epoch, validation=False):
        print(f"Starting epoch {epoch}, validation: {validation} " + "="*30,flush=True)

        loss_value = util.AverageMeter()
        # house keeping
        self.model.train()
        if self.lr_schedule(epoch+1) != self.lr_schedule(epoch):
            files.save_checkpoint_all(self.checkpoint_dir, self.model, args.arch,
                                      optimizer, self.L, epoch, lowest=False, save_str='pre-lr-drop')
        lr = self.lr_schedule(epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        XE = torch.nn.CrossEntropyLoss()
        for iter, (data, label, selected) in enumerate(loader):
            now = time.time()
            niter = epoch * len(loader) + iter

            # selected = selected.cuda(non_blocking=True)
            data = data.to(self.dev)
            if self.arch == 'pointTransformer':
                if self.npoints == 1024:
                    point_all = 1200
                elif self.npoints == 4096:
                    point_all = 4800
                elif self.npoints == 8192:
                    point_all = 8192
                else:
                    raise NotImplementedError()

                data = data.cuda(non_blocking=True)
                fps_idx = pointnet2_utils.furthest_point_sample(data, point_all)
                fps_idx = fps_idx[:, np.random.choice(point_all, self.npoints, False)]
                data = pointnet2_utils.gather_operation(data.transpose(1,2).contiguous(), fps_idx).transpose(1, 2).contiguous()

            elif self.arch == 'pointnet2':
                data = data.cuda(non_blocking=True)
                data = data.transpose(2,1)

            elif self.arch == 'point_transformer':
                data = data.cpu().numpy()
                data = provider.random_point_dropout(data)
                data[:,:, 0:3] = provider.random_scale_point_cloud(data[:,:, 0:3])
                data[:,:, 0:3] = provider.shift_point_cloud(data[:,:, 0:3])
                data = torch.Tensor(data)
                # data = data.cuda()


            if niter*args.batch_size >= self.optimize_times[-1]:
                ############ optimize sefl-label ############
                self.model.headcount = 1
                print('Optimizaton starting', flush=True)
                with torch.no_grad():
                    _ = self.optimize_times.pop()
                    self.optimize_labels(niter)

            data = data.to(self.dev)
            mass = data.size(0)
            ############ optimize self-supervised ############
            final = self.model(data)
            
            if self.hc == 1:
                loss = XE(final, self.L[0, selected])
            else:
                loss = torch.mean(torch.stack([XE(final[h],
                                                  self.L[h, selected]) for h in range(self.hc)]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value.update(loss.item(), mass)
            data = 0

            # some logging stuff 
            if iter % args.log_iter == 0:
                if self.writer:
                    self.writer.add_scalar('lr', self.lr_schedule(epoch), niter)

                    print(niter, " Loss: {0:.3f}".format(loss.item()), flush=True)
                    print(niter, " Freq: {0:.2f}".format(mass/(time.time() - now)), flush=True)
                    if writer:
                        self.writer.add_scalar('Loss', loss.item(), niter)
                        if iter > 0:
                            self.writer.add_scalar('Freq(Hz)', mass/(time.time() - now), niter)

        files.save_checkpoint_all(self.checkpoint_dir, self.model, args.arch,
                                  optimizer,  self.L, epoch, lowest=False)

        return {'loss': loss_value.avg}

    def optimize(self):
        """Perform full optimization."""
        first_epoch = 0
        self.model = self.model.to(self.dev)
        N = len(self.pseudo_loader.dataset)
        # optimization times (spread exponentially), can also just be linear in practice (i.e. every n-th epoch)
        self.optimize_times = [(self.num_epochs+2)*N] + \
                              ((self.num_epochs+1.01)*N*(np.linspace(0, 1, args.nopts)**2)[::-1]).tolist()

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    weight_decay=self.weight_decay,
                                    momentum=self.momentum,
                                    lr=self.lr)

        if self.checkpoint_dir is not None and self.resume:
            self.L, first_epoch = files.load_checkpoint_all(self.checkpoint_dir, self.model, optimizer)
            print('found first epoch to be', first_epoch, flush=True)
            include = [(qq/N >= first_epoch) for qq in self.optimize_times]
            self.optimize_times = (np.array(self.optimize_times)[include]).tolist()
        else:
            if os.path.isfile(self.pretrained_model):
                print("Load pretrained model")
                self.model.load_model_from_ckpt(self.pretrained_model)
            else:
                print("Training from scratch")
        print('We will optimize L at epochs:', [np.round(1.0 * t / N, 2) for t in self.optimize_times], flush=True)

        if first_epoch == 0:
            # initiate labels as shuffled.
            self.L = np.zeros((self.hc, N), dtype=np.int32)
            for nh in range(self.hc):
                for _i in range(N):
                    self.L[nh, _i] = _i % self.outs[nh]
                self.L[nh] = np.random.permutation(self.L[nh])
            self.L = torch.LongTensor(self.L).to(self.dev)

        # Perform optmization 
        lowest_loss = 1e9
        epoch = first_epoch
        while epoch < (self.num_epochs+1):
            m = self.optimize_epoch(optimizer, self.train_loader, epoch,
                                    validation=False)
            if m['loss'] < lowest_loss:
                lowest_loss = m['loss']
                files.save_checkpoint_all(self.checkpoint_dir, self.model, args.arch,
                                          optimizer, self.L, epoch, lowest=True)
            epoch += 1
        print(f"optimization completed. Saving model to {os.path.join(self.checkpoint_dir,'model_final.pth.tar')}")
        torch.save(self.model, os.path.join(self.checkpoint_dir, 'model_final.pth.tar'))
        return self.model


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Self-Label')
    # optimizer
    parser.add_argument('--epochs', default=600, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size (default: 256)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate (default: 0.05)')
    parser.add_argument('--lrdrop', default=200, type=int, help='multiply LR by 0.1 every (default: 150 epochs)')
    parser.add_argument('--wd', default=-4, type=float, help='weight decay pow (default: (-5)')
    parser.add_argument('--dtype', default='f64',choices=['f64','f32'], type=str, help='SK-algo dtype (default: f64)')


    # SK algo
    parser.add_argument('--nopts', default=100, type=int, help='number of pseudo-opts (default: 100)')
    parser.add_argument('--augs', default=3, type=int, help='augmentation level (default: 3)')
    parser.add_argument('--lamb', default=25, type=int, help='for pseudoopt: lambda (default:25) ')
    parser.add_argument('--cpu', default=False, action='store_true', help='use CPU variant (slow) (default: off)')


    # architecture
    parser.add_argument('--arch', default='point_transformer', type=str, help='point_transformer or pointnet2 (default: pointnet2)')
    parser.add_argument('--ncl', default=50, type=int, help='number of clusters per head (default: 50)')
    parser.add_argument('--hc', default=1, type=int, help='number of heads (default: 1)')
    parser.add_argument('--npoints', default=2048, type=int, help='number of points for each point cloud')


    # dataset
    parser.add_argument('--data_path', default='data/sl3d_data/modelnet40_normal_resampled', help='path to folder that contains `train` and `val`', type=str)
    parser.add_argument('--dataset', default='modelnet40', type=str, help='scannet or modelnet40')
    parser.add_argument('--split', default=['trainval'], type=str)


    # housekeeping
    parser.add_argument('--device', default='0,1,2,3', type=str, help='GPU devices to use for storage and model')
    parser.add_argument('--modeldevice', default='0,1,2,3', type=str, help='GPU numbers on which the CNN runs')
    parser.add_argument('--exp', default='self-label-default/checkpoint_0', help='path to experiment directory')
    parser.add_argument('--workers', default=6, type=int,help='number workers (default: 6)')
    parser.add_argument('--comment', default='self-label-default', type=str, help='name for tensorboardX')
    parser.add_argument('--log-intv', default=1, type=int, help='save stuff every x epochs (default: 1)')
    parser.add_argument('--log-iter', default=200, type=int, help='log every x-th batch (default: 200)')
    parser.add_argument('--pretrained_model', default='', help='path to pretrained model directory')
    parser.add_argument('--pseudo_labels_path', default='', type=str, help='path to pretrained self-spervised model')
    parser.add_argument('--evaluation', default=False, type=bool)
    parser.add_argument('--supervised', default=False, type=bool)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()
    name = "%s" % args.comment.replace('/', '_')
    try:
        args.device = [int(item) for item in args.device.split(',')]
    except AttributeError:
        args.device = [int(args.device)]
    args.modeldevice = args.device
    util.setup_runtime(seed=42, cuda_dev_id=list(np.unique(args.modeldevice + args.device))) # Douglas adams's choice
    print(args, flush=True)
    print()
    print(name, flush=True)
    time.sleep(5)

    writer = SummaryWriter('./runs/%s'%name)
    writer.add_text('args', " \n".join(['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))

    # Setup model and train_loader
    model, train_loader = return_model_loader(args)
    print(len(train_loader.dataset))
    model.to('cuda:0')
    if torch.cuda.device_count() > 1:
        print("Let's use", len(args.modeldevice), "GPUs for the model")
        if len(args.modeldevice) == 1:
            print('single GPU model')
        else:
            model = nn.DataParallel(model, device_ids=list(range(len(args.modeldevice))))
    # Setup optimizer
    o = Optimizer(m=model, hc=args.hc, ncl=args.ncl, t_loader=train_loader,
                  n_epochs=args.epochs, lr=args.lr, weight_decay=10**args.wd,
                  ckpt_dir=args.exp, pretrained_model=args.pretrained_model, arch=args.arch, npoints=args.npoints)
    o.writer = writer
    # Optimize
    o.optimize()
