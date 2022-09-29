import os
import sys
from pathlib import Path

import torch
import numpy as np
import h5py
import glob
import open3d as o3d
from plyfile import PlyData, PlyElement

from models import pointnet2, point_transformer

'''
GLOBAL VARIABLE CONFIG
'''
SCANNET_ALIGN_SEM_SEG_FOLDER = 'scans_align_sem'
SL3D_LABEL_FOLDER = './sl3d_data/SL3D_labels_800_class_ckpt53_v2'
NUM_MAX_PTS = 100000

label_dict = {"2": [0, 3, 17, 25, 28, 35, 59, 69, 110, 119, 120, 122, 127, 134, 138, 139, 144, 159, 181, 183, 206, 208, 237, 238, 242, 251, 257, 263, 265, 272, 304, 313, 314, 316, 318, 340, 341, 342, 358, 395, 415, 418, 424, 437, 447, 449, 467, 491, 493, 495, 499, 519, 527, 541, 543, 548, 552, 561, 571, 574, 592, 597, 599, 618, 631, 635, 641, 683, 708, 716, 717, 719, 724, 735, 739, 746], 
              "3": [77, 109, 113, 309, 344, 370, 383, 516, 689, 726], 
              "4": [1, 2, 7, 9, 11, 15, 24, 27, 30, 31, 33, 36, 38, 43, 44, 46, 51, 55, 56, 58, 60, 61, 62, 66, 68, 74, 75, 78, 79, 85, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 101, 104, 112, 114, 116, 123, 129, 130, 132, 133, 135, 136, 137, 142, 143, 145, 155, 161, 162, 164, 165, 168, 169, 174, 179, 180, 184, 185, 187, 191, 193, 195, 196, 198, 200, 202, 203, 205, 212, 215, 217, 219, 222, 228, 229, 230, 243, 244, 246, 247, 252, 256, 259, 260, 261, 266, 269, 273, 274, 282, 283, 286, 288, 289, 292, 294, 296, 298, 300, 303, 306, 308, 311, 315, 319, 324, 329, 332, 333, 336, 347, 348, 354, 356, 359, 361, 362, 366, 367, 376, 379, 382, 386, 387, 393, 398, 400, 403, 404, 406, 410, 413, 419, 421, 427, 428, 429, 434, 441, 442, 444, 446, 450, 451, 452, 453, 457, 458, 459, 465, 469, 471, 473, 478, 481, 487, 488, 498, 503, 510, 511, 515, 521, 522, 524, 525, 526, 529, 530, 531, 532, 533, 535, 538, 547, 550, 558, 559, 562, 565, 567, 579, 581, 585, 590, 591, 593, 594, 600, 601, 604, 605, 607, 612, 615, 616, 617, 621, 624, 625, 626, 629, 632, 634, 636, 640, 642, 649, 651, 653, 655, 659, 663, 671, 672, 673, 675, 677, 679, 681, 687, 688, 690, 696, 698, 699, 702, 703, 704, 706, 712, 714, 718, 722, 729, 737, 740, 741, 744, 747, 748, 750, 751, 753, 755, 758, 759, 762, 763, 764, 766, 768, 772, 773, 775, 776, 777, 782, 784, 787, 788, 789, 790, 799], 
              "5": [47, 157, 170, 310, 378, 381, 502, 505, 613, 654, 662, 685, 686, 705, 767, 779], 
              "6": [4, 20, 39, 40, 52, 71, 72, 80, 99, 103, 107, 111, 124, 141, 149, 177, 197, 232, 250, 254, 277, 278, 287, 326, 327, 349, 352, 360, 371, 384, 388, 411, 422, 480, 484, 490, 494, 496, 517, 523, 539, 544, 553, 556, 560, 570, 573, 577, 595, 603, 614, 639, 647, 658, 665, 674, 697, 700, 713, 720, 730, 731, 757, 785], 
              "7": [10, 13, 19, 21, 22, 26, 34, 45, 54, 63, 64, 76, 81, 84, 86, 93, 102, 125, 126, 131, 148, 150, 152, 158, 166, 182, 188, 190, 194, 199, 207, 209, 211, 213, 225, 233, 234, 245, 248, 249, 264, 267, 279, 280, 281, 284, 285, 290, 299, 301, 302, 305, 317, 323, 331, 338, 339, 357, 368, 373, 380, 389, 392, 405, 408, 414, 416, 417, 420, 423, 426, 431, 433, 438, 448, 454, 456, 460, 461, 475, 476, 486, 492, 497, 500, 501, 507, 512, 513, 528, 534, 537, 540, 549, 554, 555, 557, 563, 566, 575, 580, 582, 583, 587, 589, 596, 602, 610, 620, 623, 633, 637, 643, 656, 666, 680, 682, 693, 694, 701, 709, 711, 727, 738, 743, 781, 786, 791, 792, 795, 797], 
              "8": [14, 42, 48, 73, 82, 118, 146, 220, 223, 236, 239, 240, 271, 276, 291, 322, 372, 374, 375, 397, 436, 445, 470, 472, 485, 508, 514, 542, 551, 622, 644, 650, 667, 678, 684, 695, 710, 734, 771, 774, 794], 
              "9": [147, 173, 189, 227, 255, 320, 335, 462, 479, 619, 630], 
              "10": [83, 87, 105, 108, 167, 171, 186, 218, 231, 241, 275, 307, 312, 330, 351, 363, 390, 407, 409, 520, 586, 611, 723, 749, 778, 793, 798], 
              "11": [18, 67, 100, 121, 226, 235, 401, 455, 482, 489, 564, 569, 796], 
              "12": [6, 32, 50, 65, 325, 328, 337, 355, 463, 584, 598, 627, 638, 648, 657, 728], 
              "13": [435, 568, 733], 
              "14": [214], 
              "15": [345, 652, 692], 
              "16": [8, 153, 201, 402, 546, 628], 
              "17": [353, 396, 439, 536, 576, 732, 765], 
              "18": [37, 57, 115, 117, 151, 160, 224, 293, 364, 394, 399, 504, 588, 668, 707, 715, 769], 
              "19": [5, 12, 16, 23, 29, 41, 49, 53, 70, 106, 128, 140, 154, 156, 163, 172, 175, 176, 178, 192, 204, 210, 216, 221, 253, 258, 262, 268, 270, 295, 297, 321, 334, 343, 346, 350, 365, 369, 377, 385, 391, 412, 425, 430, 432, 440, 443, 464, 466, 468, 474, 477, 483, 506, 509, 518, 545, 572, 578, 606, 608, 609, 645, 646, 660, 661, 664, 669, 670, 676, 691, 721, 725, 736, 742, 745, 752, 754, 756, 760, 761, 770, 780, 783]
              }



def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= (lens + 1e-8)
    arr[:,1] /= (lens + 1e-8)
    arr[:,2] /= (lens + 1e-8)                
    return arr


def compute_normal(vertices, faces):
    #Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    normals = np.zeros( vertices.shape, dtype=vertices.dtype )
    #Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle             
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices, 
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle, 
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    normals[ faces[:,0] ] += n
    normals[ faces[:,1] ] += n
    normals[ faces[:,2] ] += n
    normalize_v3(normals)
    
    return normals


def read_ply_xyzrgbnormal(filename, real_color):
    """ read XYZ RGB normals point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 9], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = real_color[:,0] # Red
        vertices[:,4] = real_color[:,1] # Green
        vertices[:,5] = real_color[:,2] # Blue

        # compute normals
        xyz = np.array([[x, y, z] for x, y, z, _, _, _, _ in plydata["vertex"].data])
        face = np.array([f[0] for f in plydata["face"].data])
        nxnynz = compute_normal(xyz, face)
        vertices[:,6:] = nxnynz
    return vertices


def generate_sem_label():
    TRAIN_SCAN_NAMES = [line.rstrip() for line in open('meta_data/scannetv2_trainval.txt')]
    TRAIN_SL3D_FILES = [line.rstrip() for line in open('sl3d_data/train_test_files/trainval_files_gt_h5_trainval.txt')]
    
    gss_for_each_scene = [] #[[]] * len(TRAIN_SCAN_NAMES[26:])

    # Load pseudo labels
    print("Load self-labeling model")
    
    # model = pointnet2(num_classes=[800], supervised=False, evaluation=False)
    model = point_transformer(num_classes=[800], supervised=False, evaluation=False)
    ckpt = torch.load('self-label-default/checkpoint_53/lowest_299.pth')
    model = torch.nn.DataParallel(model)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    pseudo_labels = ckpt['L'][0]
    pseudo_labels = pseudo_labels.cpu()
    pseudo_labels = pseudo_labels.numpy()
    
    print("Pseudo labels generated")

    # Seperate each data w.r.s.t scene id
    for i, scan_name in enumerate(TRAIN_SCAN_NAMES[26:]):
        gss_for_each_scene.append([])
        for j, gss in enumerate(TRAIN_SL3D_FILES):
            filename = Path(gss).parts[-1]
            filename = filename[5:-3]
            if filename == scan_name:
                gss_for_each_scene[i].append([gss, pseudo_labels[j]])
    
    print("Generate semantic pseudo labels...")
    
    for idx, scan_name in enumerate(TRAIN_SCAN_NAMES[26:]):
        try:
            scene_path = './sl3d_data/scannet_all_points/' + scan_name
            mesh_fn = np.load(scene_path + '_vert.npy')
            color_mesh = mesh_fn[:, 3:6]
            semseg = os.path.join(SCANNET_ALIGN_SEM_SEG_FOLDER, scan_name + '_sem-seg.ply')
            gss_files = gss_for_each_scene[idx]

            pcd = read_ply_xyzrgbnormal(semseg, color_mesh)
            sem_label = np.load(scene_path + '_sem_label.npy')
            indices_floor = [np.where(sem_label == 0)][0]
            indices_wall = [np.where(sem_label == 1)][0]
            pcd_points = pcd[:,0:3]

            list_of_pcd_points_index = list(range(0, len(pcd_points)))
            instance_bboxes = np.zeros((len(gss_files), 7))
            instance_points_list = []
            instance_labels_list = []
            semantic_labels_list = []
            
            for gss_index, gss in enumerate(gss_files):
                gss_file_names = gss[0]
            
                gss_label = [k for k, v in label_dict.items() if gss[1] in v][0]
                
                f = h5py.File(gss_file_names[7:], 'r')
                box = np.asarray(f['prop'])
                x_min = (2 * box[0] - box[3]) / 2
                x_max = (2 * box[0] + box[3]) / 2
                y_min = (2 * box[1] - box[4]) / 2
                y_max = (2 * box[1] + box[4]) / 2
                z_min = (2 * box[2] - box[5]) / 2
                z_max = (2 * box[2] + box[5]) / 2

                box_min = np.array([x_min, y_min, z_min])
                box_max = np.array([x_max, y_max, z_max])

                bbox = np.array([(x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2,
                                x_max - x_min, y_max - y_min, z_max - z_min, gss_label])
                
                instance_bboxes[gss_index,:] = bbox

                gather_points_for_point_index = []

                for actual_index, point_index in enumerate(list_of_pcd_points_index):
                    check_1 = pcd_points[point_index] > box_min
                    check_2 = pcd_points[point_index] < box_max

                    if np.all([check_1, check_2]) == True:
                        list_of_pcd_points_index.pop(actual_index)
                        instance_points = pcd[point_index, :]
                        gather_points_for_point_index.append(instance_points)
                gather_points_for_point_index = np.array(gather_points_for_point_index)
                instance_points_list.append(gather_points_for_point_index)
                instance_labels_list.append(np.ones((gather_points_for_point_index.shape[0], 1)) * gss_index)
                semantic_labels_list.append(np.ones((gather_points_for_point_index.shape[0], 1)) * int(gss_label))
            
            scene_points = np.concatenate(instance_points_list, 0)
            scene_points = scene_points[:,0:9] # XYZ+RGB+NORMAL
            instance_labels = np.concatenate(instance_labels_list, 0) 
            semantic_labels = np.concatenate(semantic_labels_list, 0)
            data = np.concatenate((scene_points, instance_labels, semantic_labels), 1)

            # Add wall and floor semseg
            pcd_floor = np.c_[pcd[indices_floor], np.full(len(pcd[indices_floor]), gss_index + 1), np.full(len(pcd[indices_floor]), 0)]
            pcd_wall = np.c_[pcd[indices_wall], np.full(len(pcd[indices_wall]), gss_index + 2), np.full(len(pcd[indices_wall]), 1)]

            data = np.vstack((data, pcd_floor, pcd_wall))

            if data.shape[0] > NUM_MAX_PTS:
                choices = np.random.choice(data.shape[0], NUM_MAX_PTS, replace=False)
                data = data[choices]

            file_name = SL3D_LABEL_FOLDER + '/' +  scan_name + '.npy'
            bbox_file_name = SL3D_LABEL_FOLDER + '/' + scan_name +'_bbox.npy'
            with open(file_name, 'wb') as f:
                np.save(f, data)
            
            with open(bbox_file_name, 'wb') as g:
                np.save(g, instance_bboxes)
            print(" ")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("shape of subsampled scene data: {}".format(data.shape))
            print("Pseudo labels (bbox, cls, instance, semseg) for: " + scan_name + " saved")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(" ")

        except:
            print(" ")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("Failed to produce pseudo labels: " + scan_name)
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(" ")

            
if __name__ == '__main__':
    generate_sem_label()
