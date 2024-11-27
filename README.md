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
