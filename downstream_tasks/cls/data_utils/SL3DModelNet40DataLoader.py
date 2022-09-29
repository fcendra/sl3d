import os
import sys

import numpy as np
import warnings
import pickle
import sl3d_models
import torch

from tqdm import tqdm
from torch.utils.data import Dataset
from sl3d_models import point_transformer


warnings.filterwarnings('ignore')

# label_dict = {
#               0 : [25], # Airplane
#               1 : [36], # Bathtub
#               2 : [0], # Bed
#               3 : [3], # Bench
#               4 : [29], # Bookshelf
#               5 : [1], # Bootle
#               6 : [9], # Bowl
#               7 : [27], # Car
#               8 : [24], # Chair
#               9 : [18], # Cone
#               10 : [21], # Cup
#               11 : [32], # Curtain
#               12 : [31], # Desk
#               13 : [28], # Door
#               14 : [7], # Dresser
#               15 : [16], # Flower pot
#               16 : [26], # Glass box
#               17 : [5], # Guitar
#               18 : [8], # Keyboard
#               19 : [17], # Lamp
#               20:  [33], # Laptop
#               21 : [23], # Mantel
#               22 : [2], # Monitor
#               23 : [10], # Night stand
#               24 : [13], # Person
#               25 : [14], # Piano
#               26 : [19], # Plant
#               27 : [35], # Radio
#               28 : [15], # Range hood
#               29 : [12], # Sink
#               30 : [34], # Sofa
#               31 : [30], # Stairs
#               32 : [4], # Stool
#               33 : [22], # Table
#               34 : [37], # Tent
#               35 : [38], # Toilet
#               36 : [20], # Tv stand
#               37 : [11], # Vase
#               38 : [39], # Wardrobe
#               39 : [6], # Xbox
#              }

label_dict = {"0": [7, 33, 72, 81, 90, 94, 98, 107, 124, 145, 156, 180, 187, 208, 218, 219, 246, 253, 260, 261, 272, 280, 283, 296, 310, 320, 322, 326, 327, 332, 333, 349, 371, 401, 424, 438, 452, 467, 474, 521, 539, 541, 562, 570, 571, 576, 637, 644, 652, 656, 688, 689, 726, 766, 786, 819, 848, 856, 861, 888, 892, 912, 937, 944, 958, 971, 977, 986, 995, 996, 1088, 1098, 1103, 1135, 1143, 1170, 1176, 1204, 1217, 1218, 1232, 1235, 1248, 1256, 1262, 1299, 1320, 1341, 1344, 1358, 1363, 1368, 1411, 1424, 1437, 1447, 1449, 1468, 1471, 1567, 1580], 
              "1": [204, 215, 361, 363, 429, 440, 582, 598, 900, 950, 1055, 1112, 1171, 1236, 1419, 1477, 1493], 
              "2": [0, 1, 23, 47, 56, 102, 108, 110, 131, 134, 152, 169, 171, 196, 221, 286, 293, 323, 394, 402, 411, 433, 444, 457, 479, 503, 508, 512, 522, 526, 531, 536, 552, 556, 565, 589, 625, 662, 695, 696, 717, 719, 733, 739, 818, 836, 842, 844, 858, 934, 953, 965, 981, 1006, 1040, 1041, 1073, 1105, 1108, 1116, 1158, 1165, 1173, 1184, 1225, 1233, 1261, 1287, 1304, 1318, 1321, 1326, 1331, 1348, 1352, 1365, 1379, 1415, 1452, 1464, 1466, 1473, 1494, 1546, 1552, 1555, 1564], 
              "3": [26, 58, 70, 114, 154, 265, 351, 364, 609, 673, 906, 947, 972, 1037, 1051, 1067, 1148, 1194, 1242, 1332, 1482, 1538], 
              "4": [3, 6, 28, 53, 59, 68, 92, 113, 115, 130, 133, 160, 178, 183, 200, 241, 252, 268, 347, 358, 360, 372, 392, 406, 430, 443, 445, 484, 516, 518, 535, 542, 543, 550, 572, 574, 584, 611, 632, 671, 680, 698, 701, 707, 713, 721, 742, 762, 778, 805, 830, 860, 874, 881, 886, 890, 895, 904, 916, 938, 946, 960, 974, 988, 1013, 1022, 1049, 1060, 1070, 1083, 1102, 1117, 1122, 1125, 1198, 1209, 1211, 1213, 1221, 1226, 1230, 1237, 1270, 1337, 1345, 1346, 1351, 1378, 1384, 1390, 1399, 1406, 1409, 1421, 1431, 1474, 1508, 1509, 1549, 1556, 1562, 1595], 
              "5": [17, 19, 77, 80, 139, 168, 186, 264, 273, 274, 312, 376, 381, 419, 427, 511, 537, 638, 723, 729, 744, 751, 775, 811, 821, 826, 829, 870, 898, 915, 956, 963, 1000, 1008, 1042, 1054, 1066, 1087, 1149, 1156, 1157, 1206, 1247, 1293, 1353, 1416, 1433, 1460, 1488, 1510, 1520, 1521, 1528, 1532, 1536], 
              "6": [122, 324, 533, 730, 814, 1057, 1253, 1336, 1366, 1459], 
              "7": [16, 121, 138, 175, 185, 210, 239, 249, 266, 290, 308, 330, 463, 524, 555, 603, 649, 651, 663, 672, 674, 692, 808, 837, 868, 876, 896, 1097, 1306, 1335, 1359, 1407, 1425, 1467, 1492, 1530], 
              "8": [4, 25, 31, 32, 37, 42, 43, 44, 45, 57, 79, 84, 86, 91, 100, 103, 105, 118, 128, 132, 150, 157, 182, 184, 197, 201, 202, 203, 222, 223, 227, 232, 270, 276, 277, 285, 287, 288, 307, 309, 337, 344, 356, 369, 373, 387, 408, 413, 425, 460, 461, 482, 485, 532, 547, 549, 557, 558, 580, 583, 592, 596, 599, 620, 636, 666, 700, 702, 727, 731, 740, 752, 754, 760, 769, 773, 794, 796, 828, 833, 838, 843, 846, 850, 851, 880, 887, 908, 927, 930, 936, 952, 955, 957, 966, 970, 978, 1025, 1059, 1061, 1077, 1080, 1091, 1095, 1109, 1114, 1118, 1120, 1121, 1142, 1150, 1152, 1159, 1163, 1193, 1200, 1208, 1223, 1257, 1260, 1273, 1280, 1285, 1300, 1322, 1329, 1347, 1391, 1394, 1395, 1408, 1418, 1426, 1442, 1446, 1453, 1475, 1476, 1489, 1495, 1515, 1522, 1523, 1537, 1544, 1548, 1579, 1583, 1584, 1587], 
              "9": [13, 41, 55, 89, 177, 297, 385, 417, 420, 437, 442, 559, 591, 610, 734, 735, 903, 1106, 1147, 1172, 1255, 1272, 1274, 1339, 1457, 1526], 
              "10": [50, 343, 380, 487, 774, 859, 951, 1084, 1263], 
              "11": [49, 289, 345, 362, 469, 480, 500, 600, 641, 809, 864, 869, 1202, 1455, 1461, 1490, 1527, 1597], 
              "12": [99, 167, 207, 224, 391, 486, 630, 631, 679, 682, 711, 736, 765, 882, 883, 914, 975, 1030, 1104, 1181, 1259, 1271, 1343, 1410, 1412, 1535], 
              "13": [149, 211, 339, 355, 359, 390, 421, 481, 551, 578, 602, 684, 1124, 1297, 1396, 1514, 1577], 
              "14": [54, 66, 75, 125, 278, 346, 357, 377, 378, 418, 472, 553, 567, 568, 597, 618, 710, 793, 817, 901, 917, 921, 987, 989, 993, 1035, 1046, 1160, 1190, 1266, 1282, 1286, 1312, 1367, 1387, 1483, 1506, 1531, 1565, 1581], 
              "15": [46, 135, 488, 581, 628, 820, 929, 1065, 1239, 1252, 1269], 
              "16": [36, 38, 93, 159, 262, 275, 294, 301, 318, 426, 475, 513, 519, 681, 756, 770, 825, 926, 933, 940, 954, 968, 1003, 1007, 1026, 1033, 1044, 1045, 1071, 1275, 1292, 1380, 1445, 1472, 1505, 1525, 1545, 1570, 1572, 1575], 
              "17": [10, 64, 104, 213, 328, 342, 366, 398, 464, 476, 490, 677, 704, 705, 749, 801, 969, 990, 1014, 1016, 1032, 1141, 1179, 1195, 1219, 1220, 1250, 1361, 1500, 1519, 1529, 1582], 
              "18": [11, 165, 279, 348, 507, 629, 699, 725, 761, 806, 863, 998, 1015, 1024, 1111, 1185, 1243, 1355, 1369, 1375, 1429, 1543, 1559], 
              "19": [120, 137, 455, 491, 546, 585, 750, 767, 1064, 1164, 1199, 1330, 1388, 1491, 1547], 
              "20": [14, 151, 174, 189, 316, 329, 338, 352, 528, 738, 757, 879, 928, 948, 1009, 1169, 1182, 1210, 1392, 1518], 
              "21": [21, 67, 88, 109, 117, 142, 193, 199, 209, 244, 284, 319, 370, 441, 458, 548, 587, 594, 627, 669, 690, 780, 782, 831, 878, 893, 932, 935, 982, 984, 994, 1020, 1036, 1052, 1053, 1086, 1188, 1238, 1249, 1301, 1340, 1350, 1374, 1383, 1402, 1465, 1498, 1569, 1573, 1596], 
              "22": [15, 39, 48, 96, 136, 140, 166, 179, 214, 229, 243, 250, 303, 321, 325, 353, 436, 459, 502, 514, 523, 530, 573, 577, 588, 593, 615, 626, 639, 645, 647, 660, 665, 722, 728, 759, 777, 841, 845, 855, 865, 891, 905, 931, 945, 976, 1004, 1031, 1034, 1056, 1062, 1090, 1110, 1137, 1138, 1140, 1187, 1215, 1222, 1234, 1244, 1264, 1288, 1307, 1309, 1316, 1356, 1370, 1376, 1420, 1422, 1428, 1441, 1454, 1481, 1507, 1539, 1576], 
              "23": [27, 73, 173, 216, 231, 236, 238, 295, 423, 495, 498, 517, 538, 616, 642, 686, 709, 748, 807, 810, 813, 816, 943, 1101, 1130, 1131, 1203, 1296, 1373, 1400, 1430, 1511, 1563, 1592], 
              "24": [71, 123, 462, 520, 866, 1174, 1364, 1557], 
              "25": [22, 51, 63, 127, 141, 164, 176, 233, 235, 254, 282, 399, 470, 527, 561, 590, 601, 605, 606, 634, 658, 659, 697, 732, 741, 763, 803, 894, 959, 1002, 1039, 1050, 1058, 1100, 1107, 1168, 1180, 1192, 1216, 1334, 1342, 1360, 1381, 1382, 1393, 1470, 1516, 1571], 
              "26": [35, 52, 129, 170, 188, 248, 317, 483, 563, 575, 579, 648, 654, 691, 715, 724, 797, 812, 827, 835, 840, 920, 961, 964, 1078, 1161, 1186, 1228, 1276, 1291, 1354, 1404, 1435, 1448, 1502], 
              "27": [30, 230, 396, 448, 772, 979, 1010, 1038, 1094, 1189, 1469, 1561], 
              "28": [116, 298, 302, 336, 374, 389, 400, 410, 623, 635, 667, 668, 779, 854, 889, 985, 1082, 1281, 1303, 1357, 1371, 1560, 1599], 
              "29": [242, 414, 471, 506, 509, 687, 918, 925, 1115, 1132, 1338, 1578, 1591], 
              "30": [24, 34, 61, 65, 74, 76, 83, 101, 106, 119, 190, 192, 194, 195, 217, 247, 267, 269, 291, 300, 311, 314, 331, 350, 395, 407, 415, 416, 422, 428, 439, 453, 454, 478, 501, 525, 564, 569, 607, 608, 619, 640, 720, 745, 753, 764, 768, 781, 784, 785, 792, 832, 873, 877, 884, 899, 919, 949, 980, 992, 997, 1021, 1023, 1068, 1075, 1076, 1089, 1093, 1119, 1136, 1139, 1178, 1191, 1197, 1201, 1207, 1214, 1224, 1251, 1265, 1268, 1283, 1290, 1305, 1308, 1311, 1314, 1315, 1319, 1323, 1397, 1398, 1423, 1432, 1434, 1450, 1463, 1478, 1499, 1504, 1540, 1542, 1550, 1588, 1593], 
              "31": [112, 225, 226, 383, 434, 489, 771, 852, 923, 1254, 1456, 1590], 
              "32": [162, 263, 271, 446, 560, 633, 789, 815, 1212, 1284, 1302, 1333], 
              "33": [20, 82, 143, 144, 146, 155, 163, 234, 237, 257, 304, 315, 340, 367, 393, 397, 409, 431, 432, 450, 494, 499, 510, 515, 622, 655, 664, 787, 795, 822, 823, 849, 853, 885, 911, 999, 1001, 1017, 1092, 1126, 1133, 1134, 1146, 1151, 1154, 1155, 1245, 1246, 1258, 1289, 1310, 1328, 1349, 1401, 1417, 1443, 1462, 1479, 1496, 1497, 1513, 1533, 1551, 1553, 1554, 1558, 1566, 1594], 
              "34": [9, 78, 148, 181, 341, 386, 534, 612, 650, 685, 1028, 1081, 1162, 1177, 1377, 1389, 1438, 1484, 1485], 
              "35": [2, 5, 12, 18, 60, 85, 87, 95, 147, 158, 172, 191, 206, 228, 365, 375, 384, 465, 497, 676, 678, 706, 708, 718, 737, 746, 747, 783, 804, 824, 847, 862, 875, 907, 913, 941, 973, 983, 1005, 1047, 1048, 1099, 1144, 1145, 1167, 1175, 1196, 1205, 1295, 1324, 1327, 1385, 1403, 1414, 1436, 1440, 1487, 1501, 1512, 1524], 
              "36": [29, 97, 126, 153, 212, 256, 281, 292, 305, 334, 335, 403, 451, 468, 496, 540, 544, 545, 554, 566, 595, 614, 617, 646, 657, 670, 755, 776, 790, 791, 871, 897, 909, 1018, 1029, 1063, 1069, 1123, 1183, 1231, 1267, 1277, 1317, 1325, 1405, 1451, 1517, 1585, 1598], 
              "37": [8, 62, 69, 198, 205, 220, 240, 245, 251, 255, 258, 259, 299, 313, 354, 368, 379, 382, 388, 405, 412, 435, 447, 456, 466, 473, 492, 493, 504, 505, 529, 604, 613, 621, 643, 653, 661, 675, 683, 693, 694, 712, 714, 716, 743, 788, 798, 800, 802, 834, 839, 857, 902, 910, 922, 924, 942, 967, 991, 1011, 1012, 1027, 1043, 1072, 1074, 1079, 1085, 1096, 1113, 1127, 1128, 1153, 1166, 1227, 1229, 1240, 1241, 1278, 1279, 1294, 1313, 1362, 1372, 1386, 1413, 1427, 1439, 1444, 1458, 1486, 1503, 1541, 1568, 1586, 1589], 
              "38": [586, 624, 703, 758, 867, 939, 962, 1019, 1129, 1480, 1534], 
              "39": [40, 111, 161, 306, 404, 449, 477, 799, 872, 1298, 1574]}


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


class SL3DModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='trainval', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category
        self.split = split

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        model = point_transformer(num_classes=[1600], supervised=False, evaluation=False)
        ckpt = torch.load('self-label-default/checkpoint_62/lowest_342.pth')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        labels = ckpt['L'][0]
        labels = labels.cpu()
        self.sl3d_labels = labels.numpy()
        for idx, i in enumerate(self.sl3d_labels):
            self.sl3d_labels[idx] = np.array([[k for k, v in label_dict.items() if i in v][0]], dtype=np.int32)
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['trainval'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_trainval.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'trainval' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]

        if split == 'trainval':
            self.datapath = self.datapath[0:9843]
            self.sl3d_labels = self.sl3d_labels[0:9843]
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
                    if split == 'test':
                        self.list_of_points, self.list_of_labels = pickle.load(f)

                    else:
                        self.list_of_points, _ = pickle.load(f)
                        self.list_of_labels = self.sl3d_labels

                    

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
        if self.split == "test":
            return point_set, label[0]
        else:
            return point_set, label

    def __getitem__(self, index):
        return self._get_item(index)


# if __name__ == '__main__':
#     data = SL3DModelNetDataLoader('/data/modelnet40_normal_resampled/', split='trainval')
#     DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
#     for point, label in DataLoader:
#         print(point.shape)
#         print(label.shape)