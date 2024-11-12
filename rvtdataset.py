import h5py
import pandas as pd
import numpy as np

# data = pd.read_csv('output/dvs_output.csv')
# with h5py.File("intersection_1_2_td.h5", "w") as h5file:
#     events_group = h5file.create_group('/events')
#     events_group.create_dataset('x',data = data['x'].values)
#     events_group.create_dataset('y',data = data['y'].values)
#     events_group.create_dataset('p',data = data['pol'].values)
#     events_group.create_dataset('t',data = data['t'].values)
#     events_group.create_dataset('height',data = np.array([600]))
#     events_group.create_dataset('width',data = np.array([800]))
    
dtype = np.dtype([
    ('t','<i8'),
    ('x','<f4'),
    ('y','<f4'),
    ('w','<f4'),
    ('h','<f4'),
    ('class_id','<u4'),    
])
bbox = np.genfromtxt('output/bbox.csv', delimiter = ',', skip_header = 1, dtype = dtype)
# structured_data = np.array(bbox, dtype=dtype)
# print(structured_data)

# bbox_array = np.array(
#     [
#         (row['t'], row['x'], row['y'], row['w'], row['h'], row['class_id'])
#         for _, row in bbox.iterrows()
#     ],
#     dtype = dtype
# )
# # bbox_array = bbox.to_numpy()
np.save("intersection_1_2_td.npy", bbox)
# bbox = np.load('intersection_1_2_td.npy')

print("FINISH")
