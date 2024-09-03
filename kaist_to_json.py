import argparse
import yaml
import xml.etree.ElementTree as ET 
import os
from os.path import join
import numpy as np
import json
from tqdm import tqdm
from utils.datasets_vid import create_dataloader_rgb_ir
from utils.general import check_img_size, colorstr
import re

VAL_VIDEO_NAMES =['set00/V002/', 'set00/V006/', 'set01/V001/', 'set02/V002/', 'set04/V001/']

def read_xml_files_to_json(image_set, img_loc, rgb=True, use_bbox_midpoint=False, opt=None):
    dataset = {}
    dataset['images'], dataset['annotations'],  dataset['categories'] = [], [], []
    
    class_map={'person':0, 'people':1, 'cyclist':2, 'person?a':3, 'person?':4}
    for key,value in class_map.items():
        dataset['categories'].append({'id': value, 'name': key})

    set_vid_img = {}
    bbox_id = id = 0
    
    # path_map = file.read().splitlines()
    if image_set =='test':
        index = np.arange(0,2252)
        path_map = dict(zip(img_loc, index))
    else:
        index = range(0,len(img_loc))
        path_map = dict(zip(img_loc, index))
    
    for file in tqdm(img_loc):
        dummy_file = '/mnt/workspace/datasets/kaist-cvpr15/sanitized_annotations/sanitized_annotations/set00_V000_I00000.txt'            
        
        if image_set == 'test'    :
            set_vid = re.search(opt.regex_search[1:], file) # note that we load different type file 'set00/V000/I00000'
            set_vid = set_vid.string[set_vid.regs[0][0]:set_vid.regs[0][1]]
        else:
            set_vid = re.search(opt.regex_search[1:], file)
            set_vid = set_vid.string[set_vid.regs[0][0]:set_vid.regs[0][1]]
            
        file_name = re.search('I0....', file)
        file_name = file_name.string[file_name.regs[0][0]:file_name.regs[0][1]]      

        dummy_file = dummy_file.replace('set00_V000', set_vid.replace('/','_'))
        dummy_file = dummy_file.replace('I00000', file_name)
        with open(dummy_file, 'r') as file:
            lines = file.readlines()
        
        
        
        
        dataset['images'].append({
                                'id': int(id),
                                'im_name': f'{set_vid}/{file_name}',
                                'height': 512,
                                'width': 640
                                }
                                 )
        
        for line in lines[1:]:
            # Split the line into components based on whitespace
            parts = line.strip().split()

            x = int(parts[1])
            y = int(parts[2])
            w = int(parts[3])
            h = int(parts[4])
            if use_bbox_midpoint:
                x += w/2
                y += h/2
                
            to_ignore = 0
                
            if h < 55:
                to_ignore = 1
            
            
            # if parts[0] != 'person':
            #     to_ignore = 1
                # import pdb; pdb.set_trace()
            if opt.ignore_high_occ and int(parts[5]) == 2:
                to_ignore = 1
                
            # Parse the parts into a structured form
            dataset['annotations'].append(
                {
                # "type": parts[0],  # Assuming the first part is the type ('person')
                'id': bbox_id,
                'image_id': int(id), 
                'category_id': 1,
                'bbox':[ x, y, w, h],
                'height': h,
                'occlusion': int(parts[5]),
                'ignore': to_ignore,
                'area':w*h,
            })
            bbox_id = bbox_id + 1
            
        id += 1
            
            
            

    # dataset['annotations'] = set_vid_img
    json_object = json.dumps(dataset, indent=4)
    if image_set == 'test':
        output = opt.json_gt_loc + f'kaist_{opt.image_set}20_lframe_{opt.lframe}_stride_{opt.temporal_stride}{"_whole" if opt.whole else ""}_sanitized{"_midframe" if opt.midframe else ""}.json'
    elif 'small' in opt.data:
        output = opt.json_gt_loc + f'kaistsmall_{opt.image_set}_lframe_{opt.lframe}_stride_{opt.temporal_stride}{"_whole" if opt.whole else ""}_sanitized{"_even_val" if opt.even_val else ""}{"_all_objects" if opt.all_objects else ""}{"_midframe" if opt.midframe else ""}{"_ignore_high_occ" if opt.ignore_high_occ else ""}.json'
    else:
        output = opt.json_gt_loc + f'kaist_{opt.image_set}_lframe_{opt.lframe}_stride_{opt.temporal_stride}{"_whole" if opt.whole else ""}_sanitized{"_even_val" if opt.even_val else ""}{"_all_objects" if opt.all_objects else ""}{"_midframe" if opt.midframe else ""}{"_ignore_high_occ" if opt.ignore_high_occ else ""}.json'
    print(output)
    with open(output, 'w') as out:
        out.write(json_object)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--weights', type=str, default='yolov5l.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='./models/transformer/yolov5l_fusion_add_FLIR_aligned.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='./data/multispectral/FLIR_aligned.yaml', help='data.yaml path')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--lframe', type=int, default=6, help='Number of Local Frames in Batch')
    parser.add_argument('--gframe', type=int, default=0, help='Number of Global Frames in Batch')
    parser.add_argument('--temporal_stride', type=int, default=3, help='Local Frames in a batch are strided by this amount')
    parser.add_argument('--regex_search', type=str, default=".set...V...", help="For kaist:'.set...V...' , For camel use:'.images...' .This helps the dataloader seperate ordered list in indivual videos for kaist use:r'.set...V...' ")
    parser.add_argument('--dataset_used', type=str, default="kaist", help='dataset used: kaist, camel,')
    parser.add_argument('--temporal_mosaic', action='store_true', help='load mosaic with temporally related sequences of images')
    parser.add_argument('--use_tadaconv', action='store_true', help='load tadaconv as feature extractor')
    parser.add_argument('--image_set', type=str, default="test", help='train, val, test')
    parser.add_argument('--json_gt_loc', type=str, default='./json_gt/')
    parser.add_argument('--midframe', action='store_true')
    parser.add_argument('--all_objects', action='store_true')
    parser.add_argument('--ignore_high_occ', action='store_true')
    opt = parser.parse_args()
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1

    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps
    
    opt.hyp = hyp
    
    opt.even_val = True if "even_val" in opt.data else False
    opt.whole = True if "whole" in opt.data else False
        
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)  # data dict
    train_path_rgb = data_dict['train_rgb']
    if not opt.whole:
        test_path_rgb = data_dict['val_rgb']
    train_path_ir = data_dict['train_ir']
    if not opt.whole:
        test_path_ir = data_dict['val_ir']
    gs = 32
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
    rank = -1
    
    if opt.image_set == 'train':
        dataloader, dataset = create_dataloader_rgb_ir(train_path_rgb, train_path_ir, imgsz, opt.batch_size, gs, opt,
                                                        opt.temporal_stride, opt.lframe, opt.gframe, opt.regex_search,
                                                        hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                                        world_size=opt.world_size, workers=opt.workers,
                                                        image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '),
                                                        dataset_used=opt.dataset_used, temporal_mosaic=opt.temporal_mosaic, 
                                                        supervision_signal='midframe' if opt.midframe else 'lastframe',
                                                        use_tadaconv=opt.use_tadaconv, sanitized=True)
    elif opt.image_set == 'val':
        testloader, dataset = create_dataloader_rgb_ir(test_path_rgb, test_path_ir, imgsz_test, opt.batch_size * 2, gs, opt,
                                                        opt.temporal_stride, opt.lframe, opt.gframe, opt.regex_search,
                                                        hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                                        world_size=opt.world_size, workers=opt.workers,
                                                        pad=0.5, prefix=colorstr('val: '),
                                                        dataset_used = opt.dataset_used, is_validation=True,
                                                        supervision_signal='midframe' if opt.midframe else 'lastframe',
                                                        use_tadaconv=opt.use_tadaconv, sanitized=True)
    elif opt.image_set == 'test':
        pass
    
    if opt.image_set == 'test':
        file = open('/mnt/workspace/datasets/kaist-cvpr15/imageSets/test-all-20.txt', 'r')
        img_loc = file.read().splitlines()
    else:
        sequences = dataset.res
        img_loc = [sequence[-2 if opt.midframe else -1] for sequence in sequences ]
        
    read_xml_files_to_json(opt.image_set, img_loc, rgb=True, use_bbox_midpoint=False, opt=opt)
                                    
