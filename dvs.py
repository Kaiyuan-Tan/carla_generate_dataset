import carla
import math
import random
import time
import queue
import numpy as np
import cv2
from pascal_voc_writer import Writer
import os
import dvs_api
import csv


client = carla.Client('localhost', 2000)
world  = client.get_world()
bp_lib = world.get_blueprint_library()

# Get the map spawn points
spawn_points = world.get_map().get_spawn_points()

# spawn vehicle
vehicle_bp =bp_lib.find('vehicle.lincoln.mkz_2020')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

# Create a queue to store and retrieve the sensor data
# image_queue = queue.Queue() 
rgb_stack = [] # try to use stack instead of queue
dvs_stack = [] 
# camera.listen(image_queue.put)

# spectator
spectator = world.get_spectator()
transform = spectator.get_transform()
location = transform.location
rotation = transform.rotation
# spectator.set_transform(carla.Transform())
# spectator.set_transform(carla.Transform(carla.Location(x=-25.698477, y=-2.615946, z=14.969325),carla.Rotation(pitch=-28.920618, yaw=135.692627, roll=0.000037)))
# spectator.set_transform(carla.Transform(carla.Location(x=-62.010975, y=1.288139, z=17.589510),carla.Rotation(pitch=-48.045647, yaw=53.329460, roll=0.000444)))
spectator.set_transform(carla.Transform(carla.Location(x=123.813690, y=3.087291, z=18.886221),carla.Rotation(pitch=-27.611475, yaw=149.086655, roll=0.000053)))

# spawn camera
camera_bp = bp_lib.find('sensor.camera.rgb')
dvs_camera_bp = bp_lib.find('sensor.camera.dvs')
raw_camera_bp = bp_lib.find('sensor.camera.dvs')

camera_init_trans = carla.Transform(carla.Location(z=0))

camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=spectator)
dvs_camera = world.spawn_actor(dvs_camera_bp, camera_init_trans, attach_to=spectator)
raw_camera = world.spawn_actor(raw_camera_bp, camera_init_trans, attach_to=spectator)

camera.listen(lambda data: rgb_stack.append(data))
dvs_camera.listen(lambda data: dvs_stack.append(dvs_api.dvs_callback_img(data)))

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

# Get the world to camera matrix
world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

# Get the attributes from the camera
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()
fov = camera_bp.get_attribute("fov").as_float()

# Calculate the camera projection matrix to project from 3D -> 2D
K = build_projection_matrix(image_w, image_h, fov)
K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

# Retrieve the first image
world.tick()

output_path = "output/"
image_path = "images/"
rgb_label_path = "rgb_labels/"
event_path = "events/"
dvs_label_path = "dvs_labels/"

if not os.path.exists(output_path + image_path):
    os.makedirs(output_path + image_path)
    print("make dir: " + output_path + image_path)
if not os.path.exists(output_path + rgb_label_path):
    os.makedirs(output_path + rgb_label_path)
    print("make dir: " + output_path + rgb_label_path)
if not os.path.exists(output_path + event_path):
    os.makedirs(output_path + event_path)
    print("make dir: " + output_path + event_path)
if not os.path.exists(output_path + dvs_label_path):
    os.makedirs(output_path + dvs_label_path)
    print("make dir: " + output_path + dvs_label_path)

dvs_output_path = "output/dvs_output.csv"
with open(dvs_output_path, mode="w",  newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y', 't', 'pol'])
    file.close()

raw_camera.listen(lambda data: dvs_api.dvs_callback_csv(data, dvs_output_path))

while True:
    # Retrieve the image
    world.tick()

    # Get the camera matrix 
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    # Save the image
    # image.save_to_disk(frame_path + '.png')

    # Initialize the exporter
    # writer = Writer(frame_path + '.png', image_w, image_h)
    bboxes = []
    bboxes_dvs = []
    for npc in world.get_actors().filter('*vehicle*'):
        # if npc.id != vehicle.id:
            bb = npc.bounding_box
            dist = npc.get_transform().location.distance(spectator.get_transform().location)
            x = npc.get_velocity().x
            y = npc.get_velocity().y
            z = npc.get_velocity().z
            velocity = (x**2+y**2+z**2)**0.5
            if dist < 70:
                forward_vec = spectator.get_transform().get_forward_vector()
                ray = npc.get_transform().location - spectator.get_transform().location
                if forward_vec.dot(ray) > 0:
                    p1 = get_image_point(bb.location, K, world_2_camera)
                    verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                    x_max = -10000
                    x_min = 10000
                    y_max = -10000
                    y_min = 10000
                    for vert in verts:
                        p = get_image_point(vert, K, world_2_camera)
                        if p[0] > x_max:
                            x_max = p[0]
                        if p[0] < x_min:
                            x_min = p[0]
                        if p[1] > y_max:
                            y_max = p[1]
                        if p[1] < y_min:
                            y_min = p[1]

                    # Add the object to the frame (ensure it is inside the image)
                    if x_min > 0 and x_max < image_w and y_min > 0 and y_max < image_h: 
                        center_x = (x_min + x_max)/2
                        center_y = (y_min + y_max)/2
                        w_normal = (x_max - x_min)/image_w
                        h_normal = (y_max - y_min)/image_h
                        x_normal = center_x/image_w
                        y_normal = center_y/image_h

                        bboxes.append(('0', x_normal, y_normal, w_normal, h_normal))
                        if velocity >=0.1:
                            bboxes_dvs.append(('0', x_normal, y_normal, w_normal, h_normal))
    for npc in world.get_actors().filter('*pedestrian*'):
        # if npc.id != vehicle.id:
            bb = npc.bounding_box
            dist = npc.get_transform().location.distance(spectator.get_transform().location)
            x = npc.get_velocity().x
            y = npc.get_velocity().y
            z = npc.get_velocity().z
            velocity = (x**2+y**2+z**2)**0.5
            if dist < 60:
                forward_vec = spectator.get_transform().get_forward_vector()
                ray = npc.get_transform().location - spectator.get_transform().location
                if forward_vec.dot(ray) > 0:
                    p1 = get_image_point(bb.location, K, world_2_camera)
                    verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                    x_max = -10000
                    x_min = 10000
                    y_max = -10000
                    y_min = 10000
                    for vert in verts:
                        p = get_image_point(vert, K, world_2_camera)
                        if p[0] > x_max:
                            x_max = p[0]
                        if p[0] < x_min:
                            x_min = p[0]
                        if p[1] > y_max:
                            y_max = p[1]
                        if p[1] < y_min:
                            y_min = p[1]

                    # Add the object to the frame (ensure it is inside the image)
                    if x_min > 0 and x_max < image_w and y_min > 0 and y_max < image_h: 
                        center_x = (x_min + x_max)/2
                        center_y = (y_min + y_max)/2
                        w_normal = (x_max - x_min)/image_w
                        h_normal = (y_max - y_min)/image_h
                        x_normal = center_x/image_w
                        y_normal = center_y/image_h

                        bboxes.append(('1', x_normal, y_normal, w_normal, h_normal))
                        # if velocity > 0:
                        bboxes_dvs.append(('1', x_normal, y_normal, w_normal, h_normal))
    # Save the bounding boxes in the scene

    world.tick()
    image = rgb_stack.pop()
    event = dvs_stack.pop()

    frame_path = '%06d' % image.frame
    image.save_to_disk(output_path + image_path + frame_path + '.png') # YOLO format
    cv2.imwrite(output_path + event_path + frame_path + '.png', event)


    with open(output_path + rgb_label_path + frame_path+".txt", "w", encoding = "utf-8") as file:
        for bbox in bboxes:
            file.write(bbox[0]+f" {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")
        file.close()
    with open(output_path + dvs_label_path + frame_path+".txt", "w", encoding = "utf-8") as file:
        for bbox in bboxes_dvs:
            file.write(bbox[0]+f" {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")
        file.close()


# cv2.destroyAllWindows()
