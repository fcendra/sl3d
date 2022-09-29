import os

from plyfile import PlyData
import numpy as np
import open3d as o3d
import h5py
import glob
from random import shuffle

from util import create_color_palette, print_error

#############################################################################
'''
GLOBAL VARIABLE CONFIG
'''
# Crop scannet 3d scene
SCANNET_ALIGN_SEM_SEG_FOLDER = 'scans_align_sem'
SCANNET_OBJECT_PLY_FOLDER = './sl3d_data/scannet_object_unsup_ply_wypr_ensambled'
SCANNET_OBJECT_CUT_FOLDER = './sl3d_data/scannet_object_unsup_h5_wypr_ensambled'

if not os.path.exists(SCANNET_ALIGN_SEM_SEG_FOLDER): os.mkdir(SCANNET_ALIGN_SEM_SEG_FOLDER)
if not os.path.exists(SCANNET_OBJECT_PLY_FOLDER): os.mkdir(SCANNET_OBJECT_PLY_FOLDER)
if not os.path.exists(SCANNET_OBJECT_CUT_FOLDER): os.mkdir(SCANNET_OBJECT_CUT_FOLDER)

#############################################################################


def randomize_files(file_list):
    shuffle(file_list)


def normalize(point_cloud):
    centroid = np.mean(point_cloud, axis=0)
    point_cloud = point_cloud - centroid
    furthest_distance = np.max(np.sqrt(np.sum(point_cloud**2, axis=-1)))

    point_cloud /= furthest_distance
    return point_cloud


def align_axis(num_verts, plydata, axis_align_matrix):
    pts = np.ones(shape=[num_verts, 4], dtype=np.float32)
    pts[:,0] = plydata['vertex'].data['x']
    pts[:,1] = plydata['vertex'].data['y']
    pts[:,2] = plydata['vertex'].data['z']
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    plydata['vertex'].data['x'] = pts[:,0]
    plydata['vertex'].data['y'] = pts[:,1]
    plydata['vertex'].data['z'] = pts[:,2]
    return plydata


def load_align_matrix(meta_file):
    """ Load scene axis alignment matrix """
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    return axis_align_matrix


def vis_sem_seg_align(labels, mesh_file, output_file, meta_file):
    if not output_file.endswith('.ply'):
        print_error('output file must be a .ply file')
    colors = create_color_palette()
    num_colors = len(colors)
    axis_align_matrix = load_align_matrix(meta_file)

    with open(mesh_file, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        if num_verts != len(labels):
            print_error('#predicted labels = ' + str(len(labels)) + 'vs #mesh vertices = ' + str(num_verts))
        plydata = align_axis(num_verts, plydata, axis_align_matrix)
        for i in range(num_verts):
            if labels[i] >= num_colors:
                print_error('found predicted label ' + str(labels[i]) + ' not in nyu40 label set')
            color = colors[labels[i]]
            plydata['vertex']['red'][i] = color[0]
            plydata['vertex']['green'][i] = color[1]
            plydata['vertex']['blue'][i] = color[2]
    plydata.write(output_file)

def random_data_cut(xyz, box, filename):
    out_path = os.path.join(SCANNET_OBJECT_CUT_FOLDER, filename)

    h5_full_path = '/xyz'
    box_path = '/prop'
    file_prop = h5py.File(out_path, 'a')
    file_prop.create_dataset(box_path, data = box)
    file_prop.create_dataset(h5_full_path, data = xyz)
    file_prop.close()

    print(out_path + " added proposal file...")


def get_one_data():
    TRAIN_SCAN_NAMES = [line.rstrip() for line in open('meta_data/scannetv2_trainval.txt')]
   
    for scan_name in TRAIN_SCAN_NAMES:
        props = []
        
        mesh_fn = os.path.join('./sl3d_data/scans', scan_name, scan_name + '_vh_clean_2.ply') # From /home/data1/scannet/scans (symlink)
        meta_fn =  os.path.join('./sl3d_data/scans', scan_name, scan_name + '.txt')
        
        # Sem-seg
        scene_name = './sl3d_data/scannet_all_points/' + scan_name # From /home/data1/wypr_data/wypr/scannet_all_points (symlink)
        semantic_labels = np.load(scene_name+'_sem_label.npy')
        output_fn = os.path.join(SCANNET_ALIGN_SEM_SEG_FOLDER, scan_name + '_sem-seg.ply') 
        vis_sem_seg_align(semantic_labels, mesh_fn, output_fn, meta_fn)

        # GSS 
        gss_proposals = os.path.join('./sl3d_data/gss_data/scannet_gss_unsup_ensambled/',scan_name + '_prop.npy')  

        gss_proposals_list = np.load(gss_proposals)

        pcd = o3d.io.read_point_cloud(output_fn)
        pcd_points = np.asarray(pcd.points)

        # Remove wall and floor pointclouds
        indices_floor = [np.where(semantic_labels == 1)]
        indices_wall = [np.where(semantic_labels == 2)]
        indices = np.concatenate((indices_floor, indices_wall), axis=None)
        processed_pcd_points = np.delete(pcd_points, indices, 0)

        # Save processed pointcloud scene
        processed_pcd = o3d.geometry.PointCloud()
        processed_pcd.points = o3d.utility.Vector3dVector(processed_pcd_points.astype(np.float32))
        # o3d.io.write_point_cloud(output_fn, processed_pcd, write_ascii=True)
        count = 0
        
        # For every proposal
        for box in gss_proposals_list:
            
            # AABB bbox
            x_min = (2 * box[0] - box[3]) / 2
            x_max = (2 * box[0] + box[3]) / 2
            y_min = (2 * box[1] - box[4]) / 2
            y_max = (2 * box[1] + box[4]) / 2
            z_min = (2 * box[2] - box[5]) / 2
            z_max = (2 * box[2] + box[5]) / 2

            # Crop object
            box_min = np.array([x_min, y_min, z_min])
            box_max = np.array([x_max, y_max, z_max])
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound = box_min, max_bound = box_max)
            oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(bbox)
            generated_object = processed_pcd.crop(oriented_bounding_box)
            xyz = np.asarray(generated_object.points)
            
            try:
                xyz = normalize(xyz)
                if (2048 <= xyz.shape[0] <= 15000) and xyz.size != 0:
                    props.append([box_min, box_max, xyz, box])

            except ValueError:  #raised if `y` is empty.
                pass
        
        for prop in props:

            filename = "{:04d}".format(count)+"_"+scan_name
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(prop[2].astype(np.float32))
            o3d.io.write_point_cloud(os.path.join(SCANNET_OBJECT_PLY_FOLDER, filename+".ply"), object_pcd, write_ascii=True)
                    
            random_data_cut(prop[2], prop[3], filename+".h5")
            count += 1


def get_train_files():
    training_list = glob.glob("./sl3d_data/scannet_object_unsup_h5_wypr_ensambled/*.h5")
    randomize_files(training_list)
    
    f = open("./sl3d_data/train_test_files/train_files_unsup_h5_wypr_ensambled.txt", "w+")
    for i in range(len(training_list)):
        f.write("./data/"+training_list[i]+"\n")
        print("saved: "+training_list[i])

    f.close()
        

if __name__ == '__main__':
    get_one_data()
    get_train_files()
