import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm
import cv2
from torch import nn

from models.experimental import attempt_load
from utils.datasets_vid import create_dataloader_rgb_ir
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized
from utils.per_class_wandb import ap_per_class_to_wandb_format
from miss_rate_and_map.evaluation_script import evaluate as evaluate_miss_rate
import copy
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import cv2
import os
import sklearn
from models.yolo_test import Model
from ptflops import get_model_complexity_info


def draw_boxes_gt(imgs, targets):
    imgs = imgs.cpu().numpy()
    if imgs.shape[0] < imgs.shape[2]:
        imgs = imgs.transpose(1, 2, 0)
    img = imgs.astype(np.uint8)  # Convert to uint8

    for x, y, w, h in targets:
        # x = x - w//2
        # y = y - h//2
        # x_ = x + w//2
        # y_ = y + h//2
        x_ = x + w
        y_ = y + h

        img = np.ascontiguousarray(img)    
        cv2.rectangle(img, (int(x), int(y)), (int(x_), int(y_)), (0, 0, 255), 2)
    return img


def draw_boxes(imgs, targets, thre=0.5):

    if imgs.shape[0] < imgs.shape[2]:
        imgs = imgs.transpose(1, 2, 0)
    img = imgs.astype(np.uint8)  # Convert to uint8

    # Filter targets for the current image in the batch
    current_targets = targets[targets[:, 4] > thre]

    for x, y, x_, y_, _, _ in current_targets:
        
        point = (int(x.round()), int(y.round())), (int(x_.round()), int(y_.round()))
        # Draw rectangle on the image
        img = np.ascontiguousarray(img)    
        cv2.rectangle(img, (int(x.round()), int(y.round())), (int(x_.round()), int(y_.round())), (255, 0, 0), 2)
    return img


def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=True,  # save auto-label confidences
         plots=False,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         is_coco=False,
         opt=None,
         path_map = None):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    
    ##############################################
    # replace all torch-10 GELU's by torch-12 GELU to solve AttributeError: 'GELU' object has no attribute 'approximate'
    ##############################################
    def torchmodify(name) :
        a=name.split('.')
        for i,s in enumerate(a) :
            if s.isnumeric() :
                a[i]="_modules['"+s+"']"
        return '.'.join(a)
    for name, module in model.named_modules() :
        if isinstance(module, nn.GELU) :
            exec('model.' + torchmodify(name)+'=nn.GELU()')
    
    # flops, params = get_model_complexity_info(model, (6, 3, 640, 640), as_strings=True, print_per_layer_stat=True)
    # print(flops)
    
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.safe_load(f)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        # if device.type != 'cpu':
        #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        print(opt.task)
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        val_path_rgb = data['val_rgb']
        val_path_ir = data['val_ir']
        if opt.use_tadaconv:
            dataloader = create_dataloader_rgb_ir(val_path_rgb, val_path_ir, imgsz, batch_size, gs, opt,
                                                opt.temporal_stride, opt.lframe, opt.gframe, opt.regex_search,
                                                pad=0.5,# rect=True, 
                                                prefix=colorstr(f'{task}: '), supervision_signal=opt.detection_head,
                                                dataset_used=opt.dataset_used, is_training=False, use_tadaconv=True, sanitized=opt.sanitized)[0]
        else:
            dataloader = create_dataloader_rgb_ir(val_path_rgb, val_path_ir, imgsz, batch_size, gs, opt,
                                                        opt.temporal_stride, opt.lframe, opt.gframe, opt.regex_search,
                                                        pad=0.5, # rect=True, 
                                                        prefix=colorstr(f'{task}: '), supervision_signal=opt.detection_head,
                                                        dataset_used=opt.dataset_used, is_training=False, use_tadaconv=False, sanitized=opt.sanitized)[0]


    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 7) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.75', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map75, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0, 0., 0.

    map50_person, map50_people, map50_cyclist, map50_person_question, \
        map75_person, map75_people, map75_cyclist, map75_person_question, \
        map_person, map_people, map_cyclist, map_person_question = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    missrate_all = 1.0 
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    jdict_conf = []
    out_dicts = {}
    image_dict = []
    boxes_dict = []
    scores_dict = []
    classes_dict = []
    class_logits_dict = []
    prob_dict = []
    img_id_dict = []
    
    gt = json.load(open('miss_rate_and_map/KAIST_annotation.json'))
    gt_dict = {img['im_name']: 
        [gt['annotations'][i]['bbox'] for i in range(len(gt['annotations'])) 
         if gt['annotations'][i]['image_id'] == img['id']] 
        for img in gt['images']}
    for batch_i, (img, targets_rgb, targets_ir, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        
        # for i in range(img.shape[0]):
        #     print("i:", i)
        #     pattern = r'/images/(set\d+)/(\w+)/(\w+)/I(\d+)\.jpg'
        #     import re
        #     match = re.search(pattern, paths[i])
        #     if match:
        #         setID = match.group(1)
        #         vid = match.group(2)
        #         frameID = match.group(4)
        #     img_ = draw_boxes_gt(img[i,[2,1,0],0,64:640-64], gt_dict[f"{setID}/{vid}/I{frameID}"])
        #     cv2.imwrite(f'logs/visual/ground_truth/{batch_i}_id{i}_{setID}_{vid}_I{frameID}.jpg', img_)
        
    # for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(testloader, desc=s)):
        # import matplotlib.pyplot as plt
        # import pdb; pdb.set_trace()
        # if batch_i < 5:
        #     continue
        # os.makedirs(f'logs/temp/DSAFull39batch_None', exist_ok=True)
        # for b in range(img.shape[0]):
        #     for frame in range(opt.lframe):
        #         cv2.imwrite(f'logs/temp/DSAFull39batch_None/image_visual_idx_{b}_frame_{frame}.png', img[b,[2, 1, 0], frame].permute(1,2,0).cpu().numpy().astype(np.uint8))
        #         cv2.imwrite(f'logs/temp/DSAFull39batch_None/image_thermal_idx_{b}_frame_{frame}.png', img[b,3:, frame].permute(1,2,0).cpu().numpy().astype(np.uint8))
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets_rgb = targets_rgb.to(device)
        targets_ir = targets_ir.to(device)
        if len(img.shape) == 5:
            nb, _, num_frames, height, width = img.shape  # batch size, channels, number_of_frames, height, width
        elif len(img.shape) == 4:
            nb, _, height, width = img.shape  # batch size, channels, height, width
        else:
            raise 

        rgb_ir_split = img.shape[1]//2

        img_rgb = img[:, :rgb_ir_split, :, :]
        img_ir = img[:, rgb_ir_split:, :, :]

        with torch.no_grad():
            
            
            # from pytorch_grad_cam import EigenCAM
            # from pytorch_grad_cam.utils.image import show_cam_on_image
            
            
            # target_layers = [model.model[-3]]
            # cam = EigenCAM(model, target_layers, use_cuda=False)
            # grayscale_cam = cam(torch.cat((img_rgb, img_ir), 1))[0, :, :]
            # # import pdb; pdb.set_trace()
            # # cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            # # Image.fromarray(cam_image)
            # for i in range(grayscale_cam.shape[-1]):
            #     cv2.imwrite(f'logs/cam/tempimg{i}.jpg', ((grayscale_cam[:,:,i])*255.).astype(np.uint8))
            # cv2.imwrite('logs/cam/tempimg_avg.jpg', ((grayscale_cam.mean(-1))*255.).astype(np.uint8))
            # import pdb; pdb.set_trace()
            
            # Run model
            t = time_synchronized()

            if opt.detector_weights == 'both':
                out = model(torch.cat((img_rgb, img_ir), 1), augment=augment)  # inference and training outputs
                out_rgb, train_out_rgb, feature_rgb = out[0]
                out_thermal, train_out_thermal, feature_thermal = out[1]
                out = torch.cat((out_rgb, out_thermal), 1)
            else:
                out, train_out, feature = model(torch.cat((img_rgb, img_ir), 1), augment=augment)  # inference and training outputs

            
            # import pdb; pdb.set_trace()
            t0 += time_synchronized() - t
            
            if opt.detection_head == 'fullframes':
                targets_split = []
                for frame in range(opt.lframe):
                    targets_frame = targets[targets[:, 1] == frame]
                    targets_frame = torch.cat([targets_frame[:, 0:1], targets_frame[:, 2:]], dim=1)
                    targets_split.append(targets_frame)
                targets = targets_split[0]
                out = out[:, 0]

            # Compute loss
            if compute_loss:
                try:
                    
                    if opt.detection_head == 'lastframe' or opt.detection_head == 'midframe':
                        if opt.detector_weights == 'both':
                            loss += compute_loss[0]([x.float() for x in train_out_rgb], targets_rgb)[1][:3]  # box, obj, cls
                            loss += compute_loss[1]([x.float() for x in train_out_thermal], targets_ir)[1][:3]  # box, obj, cls
                        else:
                            loss += compute_loss([x.float() for x in train_out], targets_rgb)[1][:3]  # box, obj, cls
                    elif opt.detection_head == 'fullframes':
                        loss_curr = 0
                        for frame in range(opt.lframe):
                            loss_curr += compute_loss([x.float()[:, frame] for x in train_out], targets_split[frame].to(device))[1][:3]  # loss scaled by batch_size
                        loss += loss_curr / opt.lframe
                    else:
                        raise Exception(f"Detection head: {opt.detection_head} is not supported.")    
                except Exception as error:
                    # handle the exception
                    print("An exception occurred:", error)
                    import pdb; pdb.set_trace()          
                

            # Run NMS
            # For test, we only care about the last frame
            targets_rgb[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets_rgb[targets_rgb[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            # out = non_max_suppression_with_conf(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=1000)
            
            
            #########################
            # Feature Visualization #
            #########################
            if opt.feature_visualization:
                feature = feature_rgb if opt.detector_weights == 'both' else feature
                # for rgb_fea_ori in feature_rgb[:1]:
                    
                    # for frame in range(rgb_fea_ori.shape[2]):
                # rgb_fea_ori = torch.cat([feature_rgb[0], 
                #                          nn.functional.interpolate(feature_rgb[1], size=(feature_rgb[0].shape[3], feature_rgb[0].shape[4]), mode='bilinear', align_corners=False),
                #                          nn.functional.interpolate(feature_rgb[2], size=(feature_rgb[0].shape[3], feature_rgb[0].shape[4]), mode='bilinear', align_corners=False)], dim=1)
                frame = 0
                print(f"visualizing batch {batch_i}, frame {frame}:", frame)
                N = 2
                # rgb_fea = rgb_fea_ori[:,:,frame].cpu().numpy()
                # B,C,H,W = rgb_fea.shape
                # import pdb; pdb.set_trace()
                rgb_fea = np.concatenate([feature[0][:,:,frame].cpu().numpy(),
                                        nn.functional.interpolate(feature[1][:,:,frame], size=(feature[0].shape[3], feature[0].shape[4]), mode='bilinear', align_corners=False).cpu().numpy(),
                                        nn.functional.interpolate(feature[2][:,:,frame], size=(feature[0].shape[3], feature[0].shape[4]), mode='bilinear', align_corners=False).cpu().numpy()], axis=1)
                # rgb_fea = rgb_fea[:, :, 10:70]
                H_, W_ = 640, 640
                B,C,H,W = rgb_fea.shape
                pca = PCA(n_components=3)
                reduced_features = pca.fit_transform(rgb_fea.transpose(0,2,3,1).reshape(B * H * W, rgb_fea.shape[1]))
                norm_features = (reduced_features - reduced_features.min()) / (reduced_features.max() - reduced_features.min())
                scale_features = sklearn.preprocessing.minmax_scale(norm_features)
                # norm_features = sklearn.preprocessing.minmax_scale(reduced_features)
                # for alpha in [1,2,3, 5, 8, 10]:
                #     exp_sclae_features = np.exp(alpha * scale_features)
                #     # norm_features = sklearn.preprocessing.minmax_scale(norm_features)
                #     exp_sclae_features = sklearn.preprocessing.minmax_scale(exp_sclae_features)
                #     exp_sclae_features = (exp_sclae_features.reshape(B, H, W, 3) * 255.).astype(np.uint8)
                #     exp_sclae_features = nn.functional.interpolate(torch.tensor(exp_sclae_features).permute(0,3,1,2), size=(H_, W_), mode='bilinear', align_corners=False).permute(0,2,3,1).cpu().numpy()
                #     for b in range(B):
                #         os.makedirs(f'logs/det_res/{opt.name}', exist_ok=True)
                #         aaa = 1
                #         imgd = cv2.addWeighted((img[b,[2,1,0],frame].permute(1,2,0).cpu().numpy()*255).astype(np.uint8), aaa, cv2.applyColorMap(exp_sclae_features[b,:,:,0], cv2.COLORMAP_JET), 1-aaa, 0)
                #         imgd = draw_boxes(imgd, out[b])
                #         cv2.imwrite(f'logs/feat_visualization/{opt.name}/feature_batch_{batch_i}_idx_{b}_frame_{frame}_dim{H}_scale_alpha{alpha}_R.png', imgd)
                #         imgd = cv2.addWeighted((img[b,[2,1,0],frame].permute(1,2,0).cpu().numpy()*255).astype(np.uint8), aaa, cv2.applyColorMap(exp_sclae_features[b,:,:,1], cv2.COLORMAP_JET), 1-aaa, 0)
                #         imgd = draw_boxes(imgd, out[b])
                #         cv2.imwrite(f'logs/feat_visualization/{opt.name}/feature_batch_{batch_i}_idx_{b}_frame_{frame}_dim{H}_scale_alpha{alpha}_B.png', imgd)
                #         imgd = cv2.addWeighted((img[b,[2,1,0],frame].permute(1,2,0).cpu().numpy()*255).astype(np.uint8), aaa, cv2.applyColorMap(exp_sclae_features[b,:,:,2], cv2.COLORMAP_JET), 1-aaa, 0)
                #         imgd = draw_boxes(imgd, out[b])
                #         cv2.imwrite(f'logs/feat_visualization/{opt.name}/feature_batch_{batch_i}_idx_{b}_frame_{frame}_dim{H}_scale_alpha{alpha}_G.png', imgd)
                #         imgd = cv2.addWeighted((img[b,[5,4,3],frame].permute(1,2,0).cpu().numpy()*255).astype(np.uint8), aaa, cv2.applyColorMap(exp_sclae_features[b,:,:,0], cv2.COLORMAP_JET), 1-aaa, 0)
                #         imgd = draw_boxes(imgd, out[b])
                #         cv2.imwrite(f'logs/feat_visualization/{opt.name}/feature_batch_{batch_i}_idx_{b}_frame_{frame}_dim{H}_scale_alpha{alpha}_R_IR.png', imgd)
                #         imgd = cv2.addWeighted((img[b,[5,4,3],frame].permute(1,2,0).cpu().numpy()*255).astype(np.uint8), aaa, cv2.applyColorMap(exp_sclae_features[b,:,:,1], cv2.COLORMAP_JET), 1-aaa, 0)
                #         imgd = draw_boxes(imgd, out[b])
                #         cv2.imwrite(f'logs/feat_visualization/{opt.name}/feature_batch_{batch_i}_idx_{b}_frame_{frame}_dim{H}_scale_alpha{alpha}_B_IR.png', imgd)
                #         imgd = cv2.addWeighted((img[b,[5,4,3],frame].permute(1,2,0).cpu().numpy()*255).astype(np.uint8), aaa, cv2.applyColorMap(exp_sclae_features[b,:,:,2], cv2.COLORMAP_JET), 1-aaa, 0)
                #         imgd = draw_boxes(imgd, out[b])
                #         cv2.imwrite(f'logs/feat_visualization/{opt.name}/feature_batch_{batch_i}_idx_{b}_frame_{frame}_dim{H}_scale_alpha{alpha}_G_IR.png', imgd)

                

                for b in range(B):
                    for thre in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
                        os.makedirs(f'logs/det_res/{opt.name}', exist_ok=True)
                        imgd = draw_boxes((img[b,[2,1,0],frame].permute(1,2,0).cpu().numpy()*255).astype(np.uint8), out[b], thre=thre)
                        cv2.imwrite(f'logs/det_res/{opt.name}/feature_batch_{batch_i}_idx_{b}_frame_{frame}_dim{H}_V_thre{thre}.png', imgd)
                        imgd = draw_boxes((img[b,[5,4,3],frame].permute(1,2,0).cpu().numpy()*255).astype(np.uint8), out[b], thre=thre)
                        cv2.imwrite(f'logs/det_res/{opt.name}/feature_batch_{batch_i}_idx_{b}_frame_{frame}_dim{H}_I_thre{thre}.png', imgd)
            # # import pdb; pdb.set_trace()
            # for i in range(len(out)):
            #     image_data = img[i, :3, 0]
            #     image = to_pil_image(image_data.cpu())

            #     fig, ax = plt.subplots(1)
            #     ax.imshow(image)

            #     for box in out[i]:
            #         x, y, x2, y2 = box[:4]  # 提取bounding box坐标
            #         rect = patches.Rectangle((x.cpu(), y.cpu()), x2.cpu() - x.cpu(), y2.cpu() - y.cpu(), linewidth=2, edgecolor='r', facecolor='none')
            #         ax.add_patch(rect)

            #     plt.axis('off')

            #     plt.savefig(f'./logs/debug/bbox_mid_{i}.jpg', bbox_inches='tight', pad_inches=0)
            #     # plt.close()
            
            
            
            # import pdb; pdb.set_trace()
            t1 += time_synchronized() - t
            # outs = []
            # for frame in range(opt.lframe):
            #     targets = targets_split[frame]
            #     targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            #     lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            #     t = time_synchronized()
            #     outs.extend(non_max_suppression(out[:, frame], conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls))
            #     t1 += time_synchronized() - t
            # out = outs
            
        # save json to Prob_en form

            
        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets_rgb[targets_rgb[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                if save_json or path_map is not None:
                    if path_map is None: # testing case, not validation
                        file = open(val_path_rgb) # visible path
                        path_map =file.read().splitlines()
                        index = np.arange(0, 2252)
                        path_map = dict(zip(path_map, index))
                image_dict.append(int(path_map[str(path)]))
                boxes_dict.append([])
                scores_dict.append([])
                classes_dict.append([])
                class_logits_dict.append([])
                prob_dict.append([])
                img_id_dict.append(int(path_map[str(path)])) 
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            # Error probably because of image dim size
            # if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
            #     if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
            #         box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
            #                      "class_id": int(cls),
            #                      "box_caption": "%s %.3f" % (names[cls], conf),
            #                      "scores": {"class_score": conf},
            #                      "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
            #         boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
            #         wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            # wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            boxes_list =  []
            scores_list = []
            classes_list = []
            class_logits_list = []
            prob_list = []

            # Append to pycocotools JSON dictionary
            if save_json or path_map is not None:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                # image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                if path_map is None: # testing case, not validation
                    file = open(val_path_rgb) # visible path
                    path_map =file.read().splitlines()
                    index = np.arange(0, 2252)
                    path_map = dict(zip(path_map, index))
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                # import pdb; pdb.set_trace()
                inverse_sigmoid = lambda x: torch.log(x / (1 - x))
                class_logits_pos, class_logits_neg = inverse_sigmoid(pred[:, 4]), inverse_sigmoid(1 - pred[:, 4])
                
                if (not opt.json_class):
                    for logits_p, logits_n, p, b in zip(class_logits_pos.tolist(), class_logits_neg.tolist(), pred.tolist(), box.tolist()):
                        jdict.append({'image_id': int(path_map[str(path)]),
                                    'category_id': 1.0,
                                    'bbox': [round(x, 5) for x in b],
                                    'prob': [round(p[4], 7)],
                                    'score': round(p[4], 7),
                                    'class_logits': [logits_p, logits_n]}),
                        if not training:
                            jdict_conf.append({'image_id': int(path_map[str(path)]),
                                        'category_id': p[5]+1,
                                        'bbox': [round(x, 5) for x in b],
                                        'prob': [round(p[4], 7)],
                                        'score': round(p[4], 7),
                                        'class_logits': [logits_p, logits_n]})
                        
                        boxes_list.append([round(x, 5) for x in b])
                        scores_list.append(round(p[4], 7))
                        classes_list.append(0)
                        class_logits_list.append([logits_p, logits_n])
                        prob_list.append([round(p[4], 7)])
       
                else:
                    # for p, b in zip(pred.tolist(), box.tolist()):
                    #     jdict.append({'image_id': int(path_map[str(path)]),
                    #                 'category_id': p[5],
                    #                 'bbox': [round(x, 5) for x in b],
                    #                 'prob': [round(p[4], 7)],
                    #                 'score': round(p[4], 7),
                    #                 'class_logits': [logits_p, logits_n]
                    #                 })             
                    assert False, "Not implemented yet" 
            
            image_dict.append(int(path_map[str(path)]))
            boxes_dict.append(boxes_list)
            scores_dict.append(scores_list)
            classes_dict.append(classes_list)
            class_logits_dict.append(class_logits_list)
            prob_dict.append(prob_list)
            img_id_dict.append(int(path_map[str(path)])) 


            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))


        
                
                
                
                
    out_dicts['image'] = image_dict
    out_dicts['boxes'] = boxes_dict
    out_dicts['scores'] = scores_dict
    out_dicts['classes'] = classes_dict
    out_dicts['image_id'] = img_id_dict
    out_dicts['class_logits'] = class_logits_dict
    out_dicts['probs'] = prob_dict
        
        
        # Plot images
        # if plots and batch_i < 3:
        #     f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
        #     Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
        #     f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
        #     Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        # print("mAP75", ap[:, 5].mean(-1))
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class

        if opt.dataset_used == 'kaist':
            map50_person, map50_people, map50_cyclist, map50_person_question, \
                    map75_person, map75_people, map75_cyclist, map75_person_question, \
                    map_person, map_people, map_cyclist, map_person_question, \
                    ap50, ap75, ap, mp, mr, map50, map75, map =  ap_per_class_to_wandb_format(nt,opt.dataset_used, p, r, ap, f1)
            if training:
                if 'small' in opt.data: # for quick debugging
                    if opt.sanitized:
                        annfile = opt.json_gt_loc + f'kaistsmall_{opt.task}_lframe_{opt.lframe}_stride_{opt.temporal_stride}_sanitized{"_even_val" if opt.even_val else ""}{"_all_objects" if opt.all_objects else ""}_ignore_high_occ.json' # needs to be generated before hand
                    else:
                        annfile = opt.json_gt_loc + f'kaistsmall_{opt.task}_lframe_{opt.lframe}_stride_{opt.temporal_stride}{"_even_val" if opt.even_val else ""}{"_all_objects" if opt.all_objects else ""}_ignore_high_occ.json' # needs to be generated before hand
                else:
                    if opt.sanitized:
                        print("sanitized")
                        annfile = opt.json_gt_loc + f'kaist_{opt.task}_lframe_{opt.lframe}_stride_{opt.temporal_stride}_sanitized{"_even_val" if opt.even_val else ""}{"_all_objects" if opt.all_objects else ""}_ignore_high_occ.json'
                    else:
                        print("original")
                        annfile = opt.json_gt_loc + f'kaist_{opt.task}_lframe_{opt.lframe}_stride_{opt.temporal_stride}{"_even_val" if opt.even_val else ""}{"_all_objects" if opt.all_objects else ""}_ignore_high_occ.json' # needs to be generated before hand
            # else:
            #     annfile = opt.json_gt_loc + 'kaist_test20.json'
                
                rstfile = copy.deepcopy(jdict)
                results = evaluate_miss_rate(annfile, rstfile, "Multispectral")
                missrate_all = results['all'].summarize(0)
        else:
            ap50, ap75, ap, mp, mr, map50, map75, map =  ap_per_class_to_wandb_format(nt,opt.dataset_used, p, r, ap, f1)

        # ap50, ap75, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        # mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(), ap.mean()
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 5  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map75, map))

    # Print results per class
    print(nc)
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap75[i], ap[i]))
            # pass

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict) and (not opt.json_class):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        sconf_thres = str(conf_thres)
        anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions_ct{sconf_thres[2:]}_iou{iou_thres}_merge.json")  # predictions json
        pred_json_conf = str(save_dir / f"{w}_predictions_ct{sconf_thres[2:]}_conf_iou{iou_thres}_merge.json")  # predictions json
        pred_json_prob_en = str(save_dir / f"{w}_predictions_ct{sconf_thres[2:]}_prob_en_iou{iou_thres}_merge.json")  # predictions json

        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        with open(pred_json_conf, 'w') as f:
            json.dump(jdict_conf, f)
            
        with open(pred_json_prob_en, 'w') as f:
            json.dump(out_dicts, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)

        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    
    if opt.dataset_used == 'kaist':
        return (mp, mr, map50, map75, map,
                map50_person, map50_people, map50_cyclist, map50_person_question,
                map75_person, map75_people, map75_cyclist, map75_person_question,
                map_person, map_people, map_cyclist, map_person_question, missrate_all,
                *(loss.cpu() / len(dataloader)).tolist()), maps, t
    else:
        return (mp, mr, map50, map75, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t

def map_image_file_to_index(sequences):
    """
    sequences: video level sequences with length lframe + gframe (gframe=0, for our applications)
    img_for_detection: last image, all the images
    """
    imgs = [sequence[-1] for sequence in sequences]
    index = np.arange(0, len(imgs))
    path_map = dict(zip(imgs, index)) # creates a map
    
    return path_map

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='/home/fqy/proj/multispectral-object-detection/best.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='./data/multispectral/FLIR_aligned.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=64, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', default=False, action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', default=False, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', default=True, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--lframe', type=int, default=6, help='Number of Local Frames in Batch')
    parser.add_argument('--gframe', type=int, default=0, help='Number of Global Frames in Batch')
    parser.add_argument('--temporal_stride', type=int, default=3, help='Local Frames in a batch are strided by this amount')
    parser.add_argument('--regex_search', type=str, default=".images...", help="For kaist:'.set...V...' , For camel use:'.images...' .This helps the dataloader seperate ordered list in indivual videos for kaist use:r'.set...V...' ")
    parser.add_argument('--dataset_used', type=str, default="kaist", help='dataset used: kaist, camel,')
    parser.add_argument('--json-class', action='store_true', help='use class number in json instead of default 1')
    parser.add_argument('--use_tadaconv', action='store_true', help='load tadaconv as feature extractor')
    parser.add_argument('--detection_head', type=str, default='lastframe', help='selects the detection head')
    # parser.add_argument('--fusion_strategy', type=str, default='GPT', help='selects the fusion strategy for thermal and RGB')
    parser.add_argument('--json_gt_loc', type=str, default='./json_gt/')
    parser.add_argument('--multiple_outputs', action='store_true', help='obtain json multiple models')
    parser.add_argument('--sanitized', action='store_true', help='using sanitized label only')
    parser.add_argument('--feature_visualization', action='store_true', help='visualize features')
    parser.add_argument('--detector_weights', type=str, default='rgb', choices=['both', 'rgb', 'thermal'], help='selects the detector weights to be used')
    
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    print(opt.data)
    check_requirements()

    # Ensure yolov_modules_to_select.yaml and parser variables are equal
    with open('./models/yolov_modules_to_select.yaml') as f:
        select_modules = yaml.safe_load(f)  # data dict
    assert opt.use_tadaconv == select_modules['use_tadaconv'], "Make sure *****use_tadaconv***** is set to either BOTH True or BOTH False in the input arguments and YAML file" 
    # assert opt.detection_head == select_modules['detection_head'], "Make sure *****detection_head***** is set same 'head' in the input arguments and YAML file"
    # assert opt.fusion_strategy == select_modules['fusion_strategy'], "Make sure *****fusion_strategy***** is set consistently in the input arguments and YAML file" 


    if opt.task in ('train', 'val', 'test'):  # run normally
        if not opt.multiple_outputs:
            test(opt.data,
                opt.weights,
                opt.batch_size,
                opt.img_size,
                opt.conf_thres,
                opt.iou_thres,
                opt.save_json,
                opt.single_cls,
                opt.augment,
                opt.verbose,
                save_txt=opt.save_txt | opt.save_hybrid,
                save_hybrid=opt.save_hybrid,
                save_conf=opt.save_conf,
                opt=opt
                )
        else:
            temp = opt.weights[0]
            for i in range(0,100):
                print('*'*40)
                # opt.weights = temp[:-8] + f'cur_{i}.pt'
                opt.weights = temp.rsplit('/', 1)[0] + f'/cur_{i}.pt'
                print(f'{opt.weights}')
                
                test(opt.data,
                opt.weights,
                opt.batch_size,
                opt.img_size,
                opt.conf_thres,
                opt.iou_thres,
                opt.save_json,
                opt.single_cls,
                opt.augment,
                opt.verbose,
                save_txt=opt.save_txt | opt.save_hybrid,
                save_hybrid=opt.save_hybrid,
                save_conf=opt.save_conf,
                opt=opt
                )
    # results, maps, times = test.test(data_dict,
    #                                  batch_size=batch_size * 2,
    #                                  imgsz=imgsz_test,
    #                                  model=ema.ema,
    #                                  single_cls=opt.single_cls,
    #                                  dataloader=testloader,
    #                                  save_dir=save_dir,
    #                                  verbose=nc < 50 and final_epoch,
    #                                  plots=plots and final_epoch,
    #                                  wandb_logger=wandb_logger,
    #                                  compute_loss=compute_loss,
    #                                  is_coco=is_coco)

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, opt=opt)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, opt=opt)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
