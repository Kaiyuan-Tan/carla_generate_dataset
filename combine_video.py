import cv2
import numpy as np

frame_rate = 10
video_filename = 'combined_output_video.avi'

video1 = cv2.VideoCapture("dvs_output_video.avi")
video2 = cv2.VideoCapture("rgb_output_video.avi")

w1 = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
h1 = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))

w2 = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))
h2 = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'XVID' codec for .avi files
video = cv2.VideoWriter(video_filename, fourcc, frame_rate, (w1+w2, h1))

while True:
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()

    if not ret1 and not ret2:
        break

    if not ret1:
        frame1 = np.zeros((h1, w1, 3), dtype=np.unit8)

    if not ret2:
        frame2 = np.zeros((h2, w2, 3), dtype=np.unit8)

    combined_frame = np.hstack((frame1, frame2))

    video.write(combined_frame)
    if cv2.waitKey(1) & 0xFF == ord("g"):
        break

video1.release()
video2.release()
video.release()
print("DONE")
cv2.destroyAllWindows()