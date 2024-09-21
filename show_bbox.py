import cv2
import xml.etree.ElementTree as ET
import os

w = 800
h = 600

def voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymin = int(float(bbox.find('ymin').text))
        ymax = int(float(bbox.find('ymax').text))
        boxes.append((name,(xmin,ymin,xmax,ymax)))
    return boxes

def yolo_txt(txt_file):
    boxes = []
    with open(txt_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().split()
            # print(data)

            # center_x = float(data[1])*w
            # center_y = float(data[2])*h

            xmin = int((float(data[1])-float(data[3])/2)*w)
            xmax = int((float(data[1])+float(data[3])/2)*w)
            ymin = int((float(data[2])-float(data[4])/2)*h)
            ymax = int((float(data[2])+float(data[4])/2)*h)
            name = data[0]
            boxes.append((name,(xmin,ymin,xmax,ymax)))
    return boxes



def draw_boxes(image_path, path):
    image = cv2.imread(image_path)
    # boxes = voc_xml(path)
    boxes = yolo_txt(path)

    for name, (xmin,ymin,xmax,ymax) in boxes:
        if(name == "0"):
            cv2.rectangle(image, (xmin,ymin),(xmax,ymax),(0,255,0),2)
        elif(name=="1"):
            cv2.rectangle(image, (xmin,ymin),(xmax,ymax),(0,0,255),2)

    cv2.imshow('bbox', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# xml_file = "/home/apg/workspace/carla_generate_dataset/output/112077.xml"
# image_path = "/home/apg/workspace/carla_generate_dataset/output/112077.png"

image_path = "/home/apg/workspace/carla_generate_dataset/output/images/000844.png"
path = "/home/apg/workspace/carla_generate_dataset/output/labels/000844.txt"

# yolo_txt(path)
draw_boxes(image_path, path)

