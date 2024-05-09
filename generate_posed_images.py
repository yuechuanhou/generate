"""
Author: Omar Alama
Generates posed images from obj files using this format:
https://github.com/autonomousvision/differentiable_volumetric_rendering/blob/master/FAQ.md
"""

import bpy
from bpy import context
import argparse
import numpy as np
import sys
import os
from pathlib import Path
from mathutils import Matrix, Vector

def BPY_CHECK(r):
    assert r == {"FINISHED"}

#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
# Source: https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
#---------------------------------------------------------------

# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in 
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
def get_calibration_matrix_K_from_blender(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
        (   0,  s_v, v_0),
        (   0,    0,   1)))
    return K

# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
        ))
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K@RT, K, RT

# ----------------------------------------------------------

def import_obj(path_to_obj):
    current_objs = set(bpy.data.objects.keys())
    BPY_CHECK(bpy.ops.wm.obj_import(filepath=os.path.abspath(path_to_obj)))
    new_objs = set(bpy.data.objects.keys())
    diff = list(new_objs - current_objs)
    assert len(diff) == 1

    obj_name = diff[0]

    return bpy.data.objects[obj_name]

def import_obj_dir(dir_path):
    obj_paths = [str(x) for x in (p.resolve() for p in Path(dir_path).glob("**/*")
                                  if p.suffix == ".obj")]
    objs = list()
    for p in obj_paths:
        objs.append(import_obj(p))
    
    override = context.copy()
    override.update({"object": objs[0], "selected_objects": objs,
                     "selected_editable_objects": objs})
    with context.temp_override(**override):
        bpy.ops.object.join()
    
    for obj_name in bpy.data.objects.keys():
        if not obj_name.startswith("_") and bpy.data.objects[obj_name].type == "MESH":
            obj = bpy.data.objects[obj_name]
            break
    else:
        raise Exception("No MESH found after merging")
    
    override = context.copy()
    override.update({"object": obj, "selected_objects": [obj],
                     "selected_editable_objects": [obj]})
    with context.temp_override(**override):
        obj.rotation_euler = [0, 0, 0]
        max_dim = max(obj.dimensions)
        obj.dimensions = obj.dimensions/max_dim
        BPY_CHECK(bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY'))
        obj.location = [0, 0, 0]
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True, properties=False)    
    
    return obj

def sample_from_dome(outer_radius, inner_radius=0, z_limit=0, origin=(0,0,0)):
    """Sample a 3D point from a the volume between two domes using a uniform distribution"""
    random_vec = np.random.rand(3)-0.5
    random_vec = random_vec / np.linalg.norm(random_vec, ord=2)
    random_len = np.random.uniform(inner_radius, outer_radius)
    random_loc = random_vec * random_len
    if random_loc[-1] < 0: # If z coordinate is negative flip it
        random_loc[-1] *= -1

    random_loc[-1] = max(random_loc[-1], z_limit) # Clamp if below z_limit FIXME This makes it not really uniformly distributed
    return random_loc + np.array(origin)

def sample_from_pos_octant(outer_radius, inner_radius=0, z_limit=0, origin=(0,0,0)):
    random_vec = np.random.rand(3)
    random_vec = random_vec / np.linalg.norm(random_vec, ord=2)
    random_len = np.random.uniform(inner_radius, outer_radius)
    random_loc = random_vec * random_len
    random_loc[-1] = max(random_loc[-1], z_limit)
    return random_loc + np.array(origin)

def setup_camera(num_frames):
    inner_radius = max(bpy.data.objects["_inner_dome"].dimensions)/2
    outer_radius = max(bpy.data.objects["_outer_dome"].dimensions)/2

    z_limit = bpy.data.objects["_z_plane"].location[-1]
    camera_matrices = dict()
    for f in range(num_frames):
        random_loc = sample_from_pos_octant(outer_radius=outer_radius, inner_radius=inner_radius, z_limit=z_limit)
        bpy.data.objects["camera"].location = random_loc
        bpy.data.objects["camera"].keyframe_insert(data_path="location", frame=f)

        random_loc = sample_from_dome(outer_radius=0.1, inner_radius=0, z_limit=0)
        random_loc[-1] = 0
        bpy.data.objects["camera_target"].location = random_loc
        bpy.data.objects["camera_target"].keyframe_insert(data_path="location", frame=f)

        bpy.context.scene.frame_set(f)
        P, K, RT = get_3x4_P_matrix_from_blender(bpy.data.objects["camera"])

        RT4x4 = np.vstack((RT, (0,0,0,1)))
        K4x4 = np.eye(4)
        K4x4[:3, :3] = K
        camera_matrices[f"world_mat_{f}"] = K4x4 @ RT4x4


    return camera_matrices

def render_scene_frames(output_dir, num_frames):
    os.makedirs(output_dir, exist_ok=True)
    scene = bpy.data.scenes["Scene"]
    scene.render.filepath = os.path.abspath(os.path.join(output_dir, "0"))
    scene.frame_start = 0
    scene.frame_end = num_frames-1
    bpy.ops.render.render(animation=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='single_template')
    parser.add_argument('--obj_dir', type=str, help="Directory from which all obj files will be loaded and combined")
    #parser.add_argument('--obj_path', type=str, default="Single obj file to import")
    parser.add_argument('--frames', type=int, default=10,
                        help='How many frames to render')
    parser.add_argument('--out_dir', type=str, default="scan_x",
                    help='Where to store')
    
    # Needed to skip blender arguments
    try:
        idx = sys.argv.index("--")
        args = sys.argv[idx+1:] # the list after '--'
    except ValueError as e: # '--' not in the list:
        args = []
    
    args = parser.parse_args(args=args)
    import_obj_dir(args.obj_dir)
    cams = setup_camera(args.frames)
    os.makedirs(args.out_dir)
    np.savez(os.path.join(args.out_dir, "cameras.npz"), **cams)
    render_scene_frames(os.path.join(args.out_dir, "image"),
                        num_frames=args.frames)