## Generate rvt and yolo dataset
### Step 1
start carla simulator

### Step 2
Run the following:
  ```bash
python generate_traffic.py -n 100 -w 60
  ```
The data will be generated in the following path:
  ```bash
carla_generate_dataset/
│
├── generate_traffic.py     # Generate Dataset
└── output/
       ├── images/              # RGB dataset - images
       │      ├── 00001.png
       │      ├── 00002.png
       │      └── ...... 
       ├── rgb_labels/          # RGB dataset - labels
       │      ├── 00001.txt
       │      ├── 00002.txt
       │      └── ...... 
       ├── bbox.csv             # Event dataset - bboxes
       └── dvs_output.csv       # Event dataset - events
  ```
### Step 3
Transform events in csv files to h5 file and npy file. RVT model use h5 file for events and npy file for bbox.
Find the python file rvtdataset.py. Before start, replace these : 
  ```
with h5py.File("<YOUR DESTINATIO FILE NAME>_td.h5", "w") as h5file: # line 6
  ```
  ```
np.save("<YOUR DESTINATIO FILE NAME>_bbox.npy", bbox) # line 35
  ```
### Step 4
Orginalize the h5 files and npy files in this format in order to run RVT preprocess script:
  ```
data_dir
  ├── test
  │     ├── ..._bbox.npy
  │     ├── ..._td.dat.h5
  │     ...
  │
  ├── train
  │     ├── ....npy
  │     ├── ..._td.dat.h5
  │     ...
  │
  └── val
        ├── ..._bbox.npy
        ├── ..._td.dat.h5
        ... 
  ```
### Step 5
Clone RVT and set up its environment. Then run the preprocess script and finally get the rvtdataset.<br>
Visit RVT repo here: [https://github.com/uzh-rpg/RVT](https://github.com/uzh-rpg/RVT)<br>
RVT has two options to preprocess dataset, Mpx and Gen1. The format of the dataset I generated belongs to Mpx.
  ```
RVT/scripts/genx/preprocess_dataset.py
  ```
