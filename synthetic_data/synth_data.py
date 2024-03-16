# 15 / 03 / 2024
# Synth Data & Coco Annotations
import blenderproc as bproc
import argparse
import os

import numpy as np
import bpy
import glob
import shutil
import mathutils
from pathlib import Path

parser = argparse.ArgumentParser(description='blender renders')
parser.add_argument('--p_scene', default='uuv/Blender/Prompts/prueba1.blend', help="Path to the scene.blend file")
parser.add_argument('--v_frames', default=None)
parser.add_argument('--v_resolution_x', type=int, default=640)
parser.add_argument('--v_resolution_y', type=int, default=480)
parser.add_argument('--v_dataset_name', required=True) #Name of the folder, required to start
args = parser.parse_args()

'''Functions ._. ,_, '''
def pos_init():  #obtaining positions of the camera through the path
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.p_scene)))

    pose_dict = {'frame': [], 'extrinsic_matrix': []}
    #num_frames = bpy.data.actions['FollowAction'].fcurves[-1].keyframe_points[-1].co
    num_frames = 60
    camera = bpy.context.scene.camera
    for frame in range(int(num_frames)):
        bpy.context.scene.frame_set(frame + 1)
        location, rotation, scale = camera.matrix_world.decompose()  # Rotations: Quaternions
        pose_dict['frame'].append(frame)
        pose_dict['extrinsic_matrix'].append(mathutils.Matrix.LocRotScale(location, rotation, scale))

    #print(pose_dict) #for testing
    return pose_dict

def object_init(): #category id for the class, starting name for identifying the object in the blender file
    objects = bproc.loader.load_blend(args.p_scene, obj_types=['mesh', 'camera'])
    for obj in objects:
        print(obj.get_name())

    [obj.set_cp('category_id', 1) for obj in objects if obj.get_name().startswith('Gun')] # set Category id and starting Name of the objects within the blender file

def camera_init():
    fx = 605.455139160156 #this have to do with the focal length, pixels and skew factor (check: https://blender.stackexchange.com/a/120063)
    fy = 604.331848144531
    cx = 325.368804931641
    cy = 248.078979492188
    k_matrix = np.array([ #method for the K matrix https://blender.stackexchange.com/a/120063
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0 ,1]
    ]) #K (Union[ndarray, Matrix]) – The 3x3 K matrix  [[fx, 0, cx],[0, fy, cy], [0, 0, 1]]
    bproc.camera.set_intrinsics_from_K_matrix(k_matrix, args.v_resolution_x, args.v_resolution_y) #(3x3 K_matrix, image_width, image_height, clip_start=None, clip_end=None)



''''''
bproc.init()
pose_dict = pos_init()
camera_init()
object_init()

output_dir = str(Path.cwd()) + '/uuv/vision/data/images/' + args.v_dataset_name #output directory. dont use str(Path.cwd().parent) in mac

if args.v_frames == None: frames = 60 #validate
else: frames = int(args.v_frames)

for frame in range(frames):
    bproc.utility.reset_keyframes()
    bproc.camera.add_camera_pose(pose_dict['extrinsic_matrix'][frame])
    data = bproc.renderer.render()
    bproc.renderer.enable_normals_output()
    seg_data = bproc.renderer.render_segmap(map_by=['instance', 'class', 'name'])  # Render segmentation data and produce instance attribute maps

    os.makedirs(f"{output_dir}", exist_ok=True) #validate
    bproc.writer.write_coco_annotations(output_dir, #(output_dir, instance_segmaps, instance_attribute_maps, colors, color_file_format='PNG', mask_encoding_format='rle', supercategory='coco_annotations', append_to_existing_output=True, jpg_quality=95, label_mapping=None, file_prefix='', indent=None)
                                        instance_segmaps = seg_data["instance_segmaps"],
                                        instance_attribute_maps = seg_data["instance_attribute_maps"],
                                        # mask_encoding_format = 'rle',
                                        colors = data["colors"],
                                        color_file_format = "JPEG",
                                        append_to_existing_output=True)