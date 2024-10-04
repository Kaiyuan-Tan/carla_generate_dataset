import os
import shutil

image_folder = "/home/apg/workspace/yolo-dataset/output15/images"
label_folder = "/home/apg/workspace/yolo-dataset/output15/labels"
destination_image_folder = "/home/apg/workspace/yolo-dataset/val/images"
destination_label_folder = "/home/apg/workspace/yolo-dataset/val/labels"

os.makedirs(destination_image_folder, exist_ok = True)
os.makedirs(destination_label_folder, exist_ok = True)

count = 0
for image_name in os.listdir(image_folder):
    file_id = image_name.split('.')[0]
    new_id = f'{count:06d}'
    label_name = file_id + ".txt"   

    image_path = os.path.join(image_folder, image_name)
    label_path = os.path.join(label_folder, label_name)

    # print(new_id)
    # print(file_id)
    count +=1

    new_image_name = new_id + ".png"
    new_label_name = new_id + '.txt'

    destination_image_path = os.path.join(destination_image_folder, new_image_name)
    destination_label_path = os.path.join(destination_label_folder, new_label_name)

    shutil.move(image_path, destination_image_path)
    shutil.move(label_path, destination_label_path)
print(count)
print("FINISH")