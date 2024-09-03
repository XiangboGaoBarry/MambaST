import os
from PIL import Image
from shutil import copyfile
import random
from tqdm import tqdm
import itertools
import warnings

def convert_tif_to_png(source_path, dest_path):
    try:
        with Image.open(source_path) as img:
            img.save(dest_path, 'PNG')
    except:
        warnings.warn(f"An image failed to convert: {source_path}")

def organize_data(src_root, dest_root):
    os.makedirs(dest_root)

    conditions = ['Day', 'Night']
    types = ['Visible', 'FIR']
    sets = ['NewTest', 'Train']
    seqs = ['FramesPos', 'FramesNeg']
    
    W, H = 640, 471

    for condition, t, set_name in tqdm(list(itertools.product(conditions, types, sets))):
        for seq in seqs:
            src_dir = os.path.join(src_root, condition, t, set_name)
            src_dir_frame = os.path.join(src_dir, seq)
            if not os.path.exists(src_dir_frame):
                continue
            annotations_dir = os.path.join(src_dir, 'Annotations')
            dst_annotation_dir = os.path.join(dest_root, 'annotation', t)
            os.makedirs(dst_annotation_dir, exist_ok=True)
            for idx, file in tqdm(enumerate(sorted(os.listdir(src_dir_frame)))):
                dst_dir = os.path.join(dest_root, set_name, condition, seq, t)
                os.makedirs(dst_dir, exist_ok=True)
                if file.endswith('.tif'):
                    src_image_path = os.path.join(src_dir_frame, file)
                    dst_image_path = os.path.join(dst_dir, f"{idx:0>5}.png")
                    convert_tif_to_png(src_image_path, dst_image_path)
                if os.path.exists(annotations_dir):
                    anno_path = os.path.join(annotations_dir, file.replace('.tif', '.txt'))
                if os.path.exists(anno_path):
                    dst_annotation_path = os.path.join(dst_annotation_dir, f"{set_name}_{condition}_{seq}_{idx:0>5}.txt")
                    with open(anno_path, 'r') as src_file:
                        with open(dst_annotation_path, 'w') as dst_file:
                            for line in src_file:
                                parts = line.strip().split()         
                                x = int(parts[0])
                                y = int(parts[1])
                                w = int(parts[2])
                                h = int(parts[3])
                                cls = int(parts[4]) - 1
                                assert cls >= 0, f"Invalid class: {cls}"

                                # 归一化xywh值
                                norm_x = x / W
                                norm_y = y / H
                                norm_w = w / W
                                norm_h = h / H

                                # 更新parts列表中对应的归一化xywh值
                                parts[0] = str(cls)
                                parts[1] = str(norm_x)
                                parts[2] = str(norm_y)
                                parts[3] = str(norm_w)
                                parts[4] = str(norm_h)

                                # 重组行并写入到目标文件
                                new_line = ' '.join(parts[:5]) + '\n'
                                dst_file.write(new_line)
                else:
                    dst_annotation_path = os.path.join(dst_annotation_dir, f"{set_name}_{condition}_{seq}_{idx:0>5}.txt")
                    open(dst_annotation_path, 'w').close()

            
            # # import pdb; pdb.set_trace()
            # os.makedirs(dst_annotation_dir, exist_ok=True)
            # if os.path.exists(annotations_dir):
            #     for file in tqdm(os.listdir(annotations_dir)):
            #         if file.endswith('.txt'):
            #             src_annotation_path = os.path.join(annotations_dir, file)
            #             dst_annotation_path = os.path.join(dst_annotation_dir, f"{set_name}_{condition}_{file}")
            #             copyfile(src_annotation_path, dst_annotation_path)


src_root = '/mnt/workspace/datasets/temp/CVC-14'
dest_root = '/mnt/workspace/datasets/temp/CVC-14-Kaist-format'

organize_data(src_root, dest_root)
