import re
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser(description='Generate train and val dataset for multispectral video object detection')
parser.add_argument('--small', action='store_true', help='Use small dataset')
parser.add_argument('-l', '--lframe', type=int, default=3, help='Number of frames in a window')
parser.add_argument('-s', '--temporal_stride', type=int, default=3, help='Temporal stride')
parser.add_argument('--midframe', action='store_true')
parser.add_argument('-e', '--even_val', action='store_true')
parser.add_argument('--whole', action='store_true', help='Use the whole dataset for training')
parser.add_argument('--local', action='store_true')
parser.add_argument('--nc', type=int, default=1, choices=[1,4], help='Number of classes')
args = parser.parse_args()

if args.even_val:
    VAL_VIDEO_NAMES =['set00/V004/', 'set01/V002/', 'set02/V004/', 'set03/V001/', 'set04/V001/', 'set05/V000/']
    SEQ_SPLIT = {'set05/V000/': 500}
else:
    VAL_VIDEO_NAMES =['set00/V002/', 'set00/V006/', 'set01/V001/', 'set02/V002/', 'set04/V001/']
    SEQ_SPLIT = {}

data_path = "/mnt/workspace/datasets/kaist-cvpr15/sanitized_annotations/sanitized_annotations"
lwindow = args.lframe * args.temporal_stride - args.temporal_stride + 1

if args.local:
    prefix = "/home"
else:
    prefix = "/mnt/workspace/datasets/kaist-cvpr15"
    

if args.even_val:
    root_dir = f"{prefix}/kaist{'small' if args.small else ''}_video_sanitized_lframe{args.lframe}_stride{args.temporal_stride}{'_whole' if args.whole else ''}_even_val" + ("_midframe" if args.midframe else "" + "_nc1" if args.nc == 1 else "")
else:
    root_dir = f"{prefix}/kaist{'small' if args.small else ''}_video_sanitized_lframe{args.lframe}_stride{args.temporal_stride}{'_whole' if args.whole else ''}" + ("_midframe" if args.midframe else "" + "_nc1" if args.nc == 1 else "")
os.makedirs(root_dir, exist_ok=True)

train_dataset = []
val_dataset = []

for filename in tqdm(sorted(os.listdir(data_path))):
    setid, vid, filename = filename.split('_')
    filename = filename.split('.')[0]
    if lwindow > int(filename[1:])+1:
        continue
    ignore = True
    with open(f"{data_path}/{setid}_{vid}_{filename}.txt", "r") as file:
        firstline = file.readline()
        line = file.readline()
        while line:
            line = line.split()
            if line[0] == "person" :
                ignore = False
            line = file.readline()
            
    if not ignore:
        if args.whole:
            train_dataset.append((f"{setid}/{vid}", filename))
        else:
            if f"{setid}/{vid}/" in VAL_VIDEO_NAMES:
                if f"{setid}/{vid}/" in SEQ_SPLIT.keys():
                    if int(filename[1:]) < SEQ_SPLIT[f"{setid}/{vid}/"]:
                        val_dataset.append((f"{setid}/{vid}", filename))
                    else:
                        train_dataset.append((f"{setid}/{vid}", filename))
                else:
                    val_dataset.append((f"{setid}/{vid}", filename))
            else:
                train_dataset.append((f"{setid}/{vid}", filename))

if args.small:
    val_dataset = val_dataset[:10]
    train_dataset = train_dataset[:50]

def write_to_file(file, dataset, modal):
    global count
    for set_vid, file_name in dataset:
        frame_num = int(file_name[1:])
        if args.midframe:
            start_idx = frame_num - lwindow//2
            end_idx = frame_num + lwindow//2
        else:
            start_idx = frame_num+1-lwindow
            end_idx = frame_num
            
        if start_idx < 0: continue
        if os.path.exists(f"/mnt/workspace/datasets/kaist-cvpr15/images/{set_vid}/{modal}/I{end_idx:05d}.jpg"):
            
            file.write(f"/mnt/workspace/datasets/kaist-cvpr15/images/{set_vid}/{modal}/{file_name}.jpg\n")


# write the train and val datasets to file
with open(f"{root_dir}/train_vis_kaist_video.txt", "w") as file:
    write_to_file(file, train_dataset, "visible")
        
with open(f"{root_dir}/train_lwir_kaist_video.txt", "w") as file:
    write_to_file(file, train_dataset, "lwir")

if not args.whole:
    with open(f"{root_dir}/val_vis_kaist_video.txt", "w") as file:
        write_to_file(file, val_dataset, "visible")
            
    with open(f"{root_dir}/val_lwir_kaist_video.txt", "w") as file:
        write_to_file(file, val_dataset, "lwir")



# Write the train and val datasets to yaml file
if args.even_val:
    yaml_file = f"./data/multispectral_temporal/kaist{'small' if args.small else ''}_video_sanitized_lframe{args.lframe}_stride{args.temporal_stride}{'_whole' if args.whole else ''}_even_val{'_midframe' if args.midframe else ''}{'_nc1' if args.nc == 1 else ''}.yaml"
else:
    yaml_file = f"./data/multispectral_temporal/kaist{'small' if args.small else ''}_video_sanitized_lframe{args.lframe}_stride{args.temporal_stride}{'_whole' if args.whole else ''}{'_midframe' if args.midframe else ''}{'_nc1' if args.nc == 1 else ''}.yaml"
with open(yaml_file, "w") as file:
    file.write(f"train_rgb: {root_dir}/train_vis_kaist_video.txt\n")
    if not args.whole:
        file.write(f"val_rgb: {root_dir}/val_vis_kaist_video.txt\n")
    file.write(f"train_ir: {root_dir}/train_lwir_kaist_video.txt\n")
    if not args.whole:
        file.write(f"val_ir: {root_dir}/val_lwir_kaist_video.txt\n")
    if args.nc == 4:
        file.write("nc: 4\n")
        file.write("names: ['person', 'people', 'cyclist', 'person?']\n")
    elif args.nc == 1: 
        file.write("nc: 1\n")
        file.write("names: ['person']\n")
    else:
        raise ValueError("nc must be either 1 or 4")
    


