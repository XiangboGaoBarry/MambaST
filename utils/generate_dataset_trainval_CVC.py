import re
from tqdm import tqdm
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Generate train and val dataset for multispectral video object detection')
parser.add_argument('--small', action='store_true', help='Use small dataset')
parser.add_argument('-l', '--lframe', type=int, default=3, help='Number of frames in a window')
parser.add_argument('-s', '--temporal_stride', type=int, default=3, help='Temporal stride')
parser.add_argument('-w', '--whole', action='store_true', help='Use the whole dataset for training')
args = parser.parse_args()

NEGTIVES = ['Train/Night/FramesNeg/', 'Train/Day/FramesNeg/']

if args.whole:
    VAL_VIDEO_NAMES = []
    SEQ_SPLIT = dict()
else:
    VAL_VIDEO_NAMES =['Train/Night/FramesPos/', 'Train/Night/FramesNeg/', 'Train/Day/FramesPos/', 'Train/Day/FramesNeg/']
    SEQ_SPLIT = {'Train/Night/FramesPos/': 277, 'Train/Night/FramesNeg/': 391, 'Train/Day/FramesPos/': 446, 'Train/Day/FramesNeg/': 292}

TEST_VIDEO_NAMES =['NewTest/Night/FramesPos/', 'NewTest/Night/FramesNeg/', 'NewTest/Day/FramesPos/', 'NewTest/Day/FramesNeg/']

prefix = "/mnt/workspace/datasets/temp/CVC-14-Kaist-format"
suffix = f"CVC14{'small' if args.small else ''}_lframe{args.lframe}_stride{args.temporal_stride}_{'whole' if args.whole else ''}"
suffix_test = f"test_CVC14{'small' if args.small else ''}_lframe{args.lframe}_stride{args.temporal_stride}"

# images_path = f"{prefix}/Train"

data_path = f"{prefix}/annotation"
lwindow = args.lframe * args.temporal_stride - args.temporal_stride + 1


root_dir = f"{prefix}/{suffix}"
os.makedirs(root_dir, exist_ok=True)

train_dataset = []
val_dataset = []
test_dataset = []

for filename in tqdm(sorted(os.listdir(data_path + "/Visible"))):
    set_name, condition, seq, idx = filename.split('_')
    idx = idx.split('.')[0]
    if lwindow > int(idx)+1:
        continue
    # ignore = True
    # with open(f"{data_path}/{set_name}_{condition}_{seq}_{idx}.txt", "r") as file:
    #     firstline = file.readline()
    #     line = file.readline()
    #     while line:
    #         line = line.split()
    #         if line[0] == "person" :
    #             ignore = False
    #         line = file.readline()
            
    video_name = f"{set_name}/{condition}/{seq}/"
    if video_name in VAL_VIDEO_NAMES:
        if video_name in SEQ_SPLIT.keys():
            if int(idx) < SEQ_SPLIT[video_name]:
                val_dataset.append((video_name, idx))
            else:
                train_dataset.append((video_name, idx))
        else:
            val_dataset.append((video_name, idx))
    elif video_name in TEST_VIDEO_NAMES:
        test_dataset.append((video_name, idx))
    else:
        train_dataset.append((video_name, idx))

if args.small:
    val_dataset = val_dataset[:10]
    train_dataset = train_dataset[:50]
    
    
def contains_annotation(anno_file):
    if not os.path.exists(f"{data_path}/Visible/{anno_file}") or not os.path.exists(f"{data_path}/FIR/{anno_file}"):
        print("Warning: Annotation file not found")
        return False
    with open(f"{data_path}/Visible/{anno_file}", 'r') as file:
        if file.readline() != "":
            return True
    with open(f"{data_path}/FIR/{anno_file}", 'r') as file:
        if file.readline() != "":
            return True
    return False
    
def write_to_file(file_vis, file_lwir, dataset):
    for video_name, idx in dataset:
        frame_num = int(idx)
        start_idx = frame_num+1-lwindow
        end_idx = frame_num+1
            
        valid = True
        if start_idx < 0: continue
        frames_to_add = np.arange(start_idx, end_idx, args.temporal_stride)
        for _frame in frames_to_add:
            if not (os.path.exists(f"{prefix}/{video_name}Visible/{_frame:05d}.png") \
                and os.path.exists(f"{prefix}/{video_name}FIR/{_frame:05d}.png")):
                valid = False
                break
        
        if video_name not in NEGTIVES:
            anno_file = video_name.replace('/', '_') + f"{idx}.txt"
            if not contains_annotation(anno_file):
                valid = False
            
            
        if valid:
            file_vis.write(f"{prefix}/{video_name}Visible/{idx}.png\n")
            file_lwir.write(f"{prefix}/{video_name}FIR/{idx}.png\n")
        # if os.path.exists(f"/mnt/workspace/datasets/kaist-cvpr15/images/{set_vid}/{modal}/I{end_idx:05d}.jpg"):
            
        #     file.write(f"/mnt/workspace/datasets/kaist-cvpr15/images/{set_vid}/{modal}/{file_name}.jpg\n")

# write the train and val datasets to file
print("train_dataset")
with open(f"{root_dir}/train_vis.txt", "w") as file_vis:
    with open(f"{root_dir}/train_lwir.txt", "w") as file_lwir:
        write_to_file(file_vis, file_lwir, train_dataset)
print("val_dataset")
with open(f"{root_dir}/val_vis.txt", "w") as file_vis:
    with open(f"{root_dir}/val_lwir.txt", "w") as file_lwir:
        write_to_file(file_vis, file_lwir, val_dataset)
print("test_dataset")
with open(f"{root_dir}/test_vis.txt", "w") as file_vis:
    with open(f"{root_dir}/test_lwir.txt", "w") as file_lwir:
        write_to_file(file_vis, file_lwir, test_dataset)



# Write the train and val datasets to yaml file
    yaml_file = f"./data/multispectral_temporal/{suffix}.yaml"
    
with open(yaml_file, "w") as file:
    file.write(f"train_rgb: {root_dir}/train_vis.txt\n")
    file.write(f"val_rgb: {root_dir}/val_vis.txt\n")
    file.write(f"train_ir: {root_dir}/train_lwir.txt\n")
    file.write(f"val_ir: {root_dir}/val_lwir.txt\n")
    file.write("nc: 1\n")
    file.write("names: ['person']\n")
    
    
# Write the train and val datasets to yaml file
    yaml_file_test = f"./data/multispectral_temporal/{suffix_test}.yaml"
    
with open(yaml_file_test, "w") as file:
    file.write(f"test_rgb: {root_dir}/test_vis.txt\n")
    file.write(f"test_ir: {root_dir}/test_lwir.txt\n")
    file.write("nc: 1\n")
    file.write("names: ['person']\n")
    


