import cv2
import pandas as pd
import numpy as np

# Paths to the CSV files containing event data and bounding boxes
data_file = 'output/dvs_output.csv'
label_file = 'output/bbox.csv'

# Path and name of the output video file
video_filename = 'dvs_output_video.avi'

# Desired frame rate of the video and frame dimensions
frame_rate = 1000
H = 600  # Height of the frame
W = 800  # Width of the frame

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'XVID' codec for .avi files
video = cv2.VideoWriter(video_filename, fourcc, frame_rate, (W, H))

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(data_file)

# Extract event data (assuming the first four columns represent [timestamp, x, y, polarity])
data = df.to_numpy()

chunk = []  # To accumulate events in chunks
for i, row in enumerate(data, start=1):
    chunk.append(row)  # Append row to chunk
    
    # Process the chunk every 100 events
    if i % 100 == 0:
        # Create a blank image (black background)
        dvs_img = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Convert the chunk to a NumPy array
        chunk_np = np.array(chunk)
        
        # Extract x, y, and polarity data
        x_coords = chunk_np[:, 0].astype(int)
        y_coords = chunk_np[:, 1].astype(int)
        polarity = chunk_np[:, 3].astype(int)
        
        # Update the image based on polarity (use red or blue for +1/-1 polarity)
        dvs_img[y_coords, x_coords, 0] = polarity * 255  # Blue channel for -1 polarity
        dvs_img[y_coords, x_coords, 2] = (1 - polarity) * 255  # Red channel for +1 polarity
        
        # Write the frame to the video
        video.write(dvs_img)
        
        # Reset the chunk for the next batch of events
        chunk = []

# In case there are remaining events in the chunk
if chunk:
    dvs_img = np.zeros((H, W, 3), dtype=np.uint8)
    chunk_np = np.array(chunk)
    x_coords = chunk_np[:, 1].astype(int)
    y_coords = chunk_np[:, 2].astype(int)
    polarity = chunk_np[:, 3].astype(int)
    
    dvs_img[y_coords, x_coords, 0] = polarity * 255  # Blue channel for -1 polarity
    dvs_img[y_coords, x_coords, 2] = (1 - polarity) * 255  # Red channel for +1 polarity
    
    # Write the last frame to the video
    video.write(dvs_img)

# Release the video writer
video.release()
cv2.destroyAllWindows()

print(f"Video saved as {video_filename}")
