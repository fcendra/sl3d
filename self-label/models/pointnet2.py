import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction

__all__ = ['Pointnet2', 'pointnet2']

class Pointnet2(nn.Module):
    def __init__(self, num_class=[50],normal_channel=False, supervised=False, evaluation=False):
        super(Pointnet2, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.supervised = supervised
        self.evaluation = evaluation
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.num_class = num_class
        self.fc3 = nn.Linear(256, 18)
        
        if self.num_class[0] > 18:
            self.fc4 = nn.Linear(256, self.num_class[0])

        self.headcount = len(self.num_class)
        if len(self.num_class) == 1:
            self.top_layer = nn.Sequential(*[nn.Linear(256, self.num_class[0])])
        else:
            for a, i in enumerate(self.num_class):
                setattr(self, "top_layer%d" % a, nn.Linear(256, i))
            self.top_layer = None


    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x)), inplace=True))
        x = self.drop2(F.relu(self.bn2(self.fc2(x)), inplace=True))

        if self.supervised == True:
            if self.num_class[0] == 18:
                x = self.fc3(x)
            else:
                x = self.fc4(x)

            x = F.log_softmax(x, -1)
            return x, l3_points
        
        elif self.evaluation == True:
            return x

        elif self.headcount == 1:
            if self.top_layer:
                x = self.top_layer(x)
            return x
        
        else:
            xp = []
            for i in range(self.headcount):
                xp.append(getattr(self, "top_layer%d" % i)(x))
            return xp


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


def pointnet2(num_classes=[50], normal_channel='store_true', supervised=False, evaluation=False):
    model = Pointnet2(num_classes, normal_channel, supervised, evaluation)
    return model


if __name__ == '__main__':
    model = pointnet2(num_classes=[100], supervised=False, evaluation=False)

