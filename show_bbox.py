import cv2
import xml.etree.ElementTree as ET
import os

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

def draw_boxes(image_path, xml_path):
    image = cv2.imread(image_path)
    boxes = voc_xml(xml_path)

    for name, (xmin,ymin,xmax,ymax) in boxes:
        cv2.rectangle(image, (xmin,ymin),(xmax,ymax),(0,255,0),2)
    
    cv2.imshow('bbox', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

xml_file = "/home/apg/workspace/carla_generate_dataset/output/112077.xml"
image_path = "/home/apg/workspace/carla_generate_dataset/output/112077.png"

draw_boxes(image_path, xml_file)

