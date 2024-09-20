import carla
import random
import csv

client = carla.Client('localhost', 2000)
world = client.get_world()