# this script reads the sanitized kasit xml files and converts them to text files

import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

target_root = "/home/labels_sanitized_tmp"
# delete the target_root directory if it exists
if os.path.exists(target_root):
    os.system(f"rm -rf {target_root}")

source_directory = '/mnt/workspace/datasets/kaist-cvpr15/annotations-xml-new-sanitized'
dataset = []
for root, dirs, files in os.walk(source_directory):
    for file in files:
        if file.endswith(".xml"):
            dataset.append((os.path.join(*root.split('/')[-2:]), file.split('.')[0]))
            
for set_vid, file_name in tqdm(sorted(dataset)):
    dummy_file = os.path.join(source_directory, 'set00/V000/I00000.xml')
    dummy_file = dummy_file.replace('set00/V000', set_vid)
    dummy_file = dummy_file.replace('I00000', file_name)
    xmlfile = dummy_file
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    annotations = []
    for obj in root.iter('object'):
        occlusion = int(obj.find('occlusion').text)
        if occlusion == 2:
            # print(f"Occlusion 2 found in {xmlfile}")
            continue
        x = int(obj.find('bndbox').find('x').text)
        y = int(obj.find('bndbox').find('y').text)
        w = int(obj.find('bndbox').find('w').text)
        h = int(obj.find('bndbox').find('h').text)
        if h*w < 50:
            # print(f"Small bounding box found in {xmlfile}")
            continue
        x = (x + w/2) / 640
        y = (y + h/2) / 512
        w = w / 640
        h = h / 512
        annotations.append('0 %.6f %.6f %.6f %.6f' % (x, y, w, h))
    os.makedirs(f'{target_root}/{set_vid}/visible', exist_ok=True)
    os.makedirs(f'{target_root}/{set_vid}/lwir', exist_ok=True)
    with open(f'{target_root}/{set_vid}/lwir/{file_name}.txt', 'w') as f:
        for ann in annotations:
            f.write(ann)
            f.write('\n')
    with open(f'{target_root}/{set_vid}/visible/{file_name}.txt', 'w') as f:
        for ann in annotations:
            f.write(ann)
            f.write('\n')
        
        
        