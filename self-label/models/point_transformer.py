import torch
import torch.nn as nn
from sl3d_models.pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction
from sl3d_models.transformer import TransformerBlock


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
    
    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2
        

class Backbone(nn.Module):
    def __init__(self, num_class=[50]):
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = 1024, 4, 16, 18, 3
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, 512, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, 512, nneighbor))
        self.nblocks = nblocks
    
    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats


class PointTransformerCls(nn.Module):
    def __init__(self, num_class=[50], supervised=False, evaluation=False):
        super().__init__()
        print("using point transformer")
        self.supervised = supervised
        self.evaluation = evaluation
        self.backbone = Backbone(num_class)
        npoints, nblocks, nneighbor, n_c, d_points = 1024, 4, 16, 18 , 3
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
        self.nblocks = nblocks
        self.num_class = num_class
        if self.num_class[0] != 18:
            self.fc3 = nn.Sequential(
                       nn.Linear(32 * 2 ** nblocks, 256),
                       nn.ReLU(),
                       nn.Linear(256, 64),
                       nn.ReLU(),
                       nn.Linear(64, self.num_class[0])
            )

        self.headcount = len(self.num_class)
        if len(self.num_class) == 1:
            self.top_layer = nn.Sequential(*[nn.Linear(512, self.num_class[0])])
        else:
            for a, i in enumerate(self.num_class):
                setattr(self, "top_layer%d" % a, nn.Linear(512, i))
            self.top_layer = None
    
    def forward(self, x):
        points, _ = self.backbone(x)
        points = points.mean(1)
        if self.supervised == True:
            if self.num_class[0] == 18:
                res = self.fc2(points)
            else:
                res = self.fc3(points)
            return res
        
        elif self.evaluation == True:
            return points
        
        elif self.headcount == 1:
            if self.top_layer:
                x = self.top_layer(points)
            return x
        
        else:
            xp = []
            for i in range(self.headcount):
                xp.append(getattr(self, "top_layer%d" % i)(points))
            return xp


def point_transformer(num_classes=[50], supervised=False, evaluation=False):
    model = PointTransformerCls(num_classes, supervised, evaluation)
    return model


if __name__ == '__main__':
    model = point_transformer(num_class=[50], supervised=False, evaluation=False)


    
