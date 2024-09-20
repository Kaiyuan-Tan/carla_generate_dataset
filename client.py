import carla
import math
import random
import time
import queue
import numpy as np
import cv2
from pascal_voc_writer import Writer


client = carla.Client('localhost', 2000)
world  = client.get_world()
bp_lib = world.get_blueprint_library()

# Get the map spawn points
spawn_points = world.get_map().get_spawn_points()

# spawn vehicle
vehicle_bp =bp_lib.find('vehicle.lincoln.mkz_2020')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

# Create a queue to store and retrieve the sensor data
image_queue = queue.Queue()
# camera.listen(image_queue.put)

# spectator
spectator = world.get_spectator()
transform = spectator.get_transform()
location = transform.location
rotation = transform.rotation
# spectator.set_transform(carla.Transform())
# spectator.set_transform(carla.Transform(carla.Location(x=-25.698477, y=-2.615946, z=14.969325),carla.Rotation(pitch=-28.920618, yaw=135.692627, roll=0.000037)))
spectator.set_transform(carla.Transform(carla.Location(x=-62.010975, y=1.288139, z=17.589510),carla.Rotation(pitch=-48.045647, yaw=53.329460, roll=0.000444)))

# spawn camera
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_init_trans = carla.Transform(carla.Location(z=0))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=spectator)

camera.listen(image_queue.put)

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

# for i in range(50):
#     vehicle_bp = random.choice(bp_lib.filter('vehicle'))
#     npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
#     if npc:
#         npc.set_autopilot(True)

# Retrieve the first image
world.tick()
image = image_queue.get()

# Reshape the raw data into an RGB array
img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)) 

# Display the image in an OpenCV display window
# cv2.namedWindow('ImageWindowName', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('ImageWindowName',img)
# cv2.waitKey(1)

# while True:

#     # Retrieve and reshape the image
#     world.tick()
#     image = image_queue.get()

#     img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

#     # Get the camera matrix 
#     world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

#     for npc in world.get_actors().filter('*vehicle*'):

#         # Filter out the ego vehicle
#         # if npc.id != vehicle.id:

#             bb = npc.bounding_box
#             dist = npc.get_transform().location.distance(spectator.get_transform().location)

#             # Filter for the vehicles within 50m
#             if dist < 60:

#             # Calculate the dot product between the forward vector
#             # of the vehicle and the vector between the vehicle
#             # and the other vehicle. We threshold this dot product
#             # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
#                 forward_vec = spectator.get_transform().get_forward_vector()
#                 ray = npc.get_transform().location - spectator.get_transform().location

#                 if forward_vec.dot(ray) > 0:
#                     p1 = get_image_point(bb.location, K, world_2_camera)
#                     verts = [v for v in bb.get_world_vertices(npc.get_transform())]
#                     x_max = -10000
#                     x_min = 10000
#                     y_max = -10000
#                     y_min = 10000

#                     for vert in verts:
#                         p = get_image_point(vert, K, world_2_camera)
#                         # Find the rightmost vertex
#                         if p[0] > x_max:
#                             x_max = p[0]
#                         # Find the leftmost vertex
#                         if p[0] < x_min:
#                             x_min = p[0]
#                         # Find the highest vertex
#                         if p[1] > y_max:
#                             y_max = p[1]
#                         # Find the lowest  vertex
#                         if p[1] < y_min:
#                             y_min = p[1]

#                     cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,0,255, 255), 1)
#                     cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
#                     cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 255), 1)
#                     cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 255), 1)

#     for npc in world.get_actors().filter('*pedestrian*'):

#             # Filter out the ego vehicle
#             # if npc.id != vehicle.id:

#                 bb = npc.bounding_box
#                 dist = npc.get_transform().location.distance(spectator.get_transform().location)

#                 # Filter for the vehicles within 50m
#                 if dist < 60:

#                 # Calculate the dot product between the forward vector
#                 # of the vehicle and the vector between the vehicle
#                 # and the other vehicle. We threshold this dot product
#                 # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
#                     forward_vec = spectator.get_transform().get_forward_vector()
#                     ray = npc.get_transform().location - spectator.get_transform().location

#                     if forward_vec.dot(ray) > 0:
#                         p1 = get_image_point(bb.location, K, world_2_camera)
#                         verts = [v for v in bb.get_world_vertices(npc.get_transform())]
#                         x_max = -10000
#                         x_min = 10000
#                         y_max = -10000
#                         y_min = 10000

#                         for vert in verts:
#                             p = get_image_point(vert, K, world_2_camera)
#                             # Find the rightmost vertex
#                             if p[0] > x_max:
#                                 x_max = p[0]
#                             # Find the leftmost vertex
#                             if p[0] < x_min:
#                                 x_min = p[0]
#                             # Find the highest vertex
#                             if p[1] > y_max:
#                                 y_max = p[1]
#                             # Find the lowest  vertex
#                             if p[1] < y_min:
#                                 y_min = p[1]

#                         cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (255,0,0, 255), 1)
#                         cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (255,0,0, 255), 1)
#                         cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (255,0,0, 255), 1)
#                         cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (255,0,0, 255), 1)

#     cv2.imshow('ImageWindowName',img)
#     if cv2.waitKey(1) == ord('q'):
#         break

while True:
    # Retrieve the image
    world.tick()
    image = image_queue.get()

    # Get the camera matrix 
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    frame_path = 'output/%06d' % image.frame

    # Save the image
    image.save_to_disk(frame_path + '.png')

    # Initialize the exporter
    writer = Writer(frame_path + '.png', image_w, image_h)

    for npc in world.get_actors().filter('*vehicle*'):
        # if npc.id != vehicle.id:
            bb = npc.bounding_box
            dist = npc.get_transform().location.distance(spectator.get_transform().location)
            if dist < 50:
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
                        writer.addObject('vehicle', x_min, y_min, x_max, y_max)

    # Save the bounding boxes in the scene
    writer.save(frame_path + '.xml')

# cv2.destroyAllWindows()
