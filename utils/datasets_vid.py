# Dataset utils and dataloaders

import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import check_requirements, xyxy2xywh, xywh2xyxy, xywhn2xyxy, xyn2xy, segment2box, segments2boxes, \
    resample_segments, clean_str
from utils.torch_utils import torch_distributed_zero_first


# import global_var
import re
from torch.utils.data.sampler import Sampler, BatchSampler
import matplotlib.pyplot as plt #for debugging
import matplotlib.patches as patches #for debugging

# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)

KAIST_ANNOTATION_PATH = "/mnt/workspace/datasets/kaist-cvpr15/sanitized_annotations/sanitized_annotations_format_all/"

# class RandomSampler(torch.utils.data.sampler.RandomSampler):

#     def __init__(self, data_source, replacement=False, num_samples=None):
#         self.data_source = data_source
#         self.replacement = replacement
#         self._num_samples = num_samples

#         if not isinstance(self.replacement, bool):
#             raise ValueError("replacement should be a boolean value, but got "
#                              "replacement={}".format(self.replacement))

#         if self._num_samples is not None and not replacement:
#             raise ValueError("With replacement=False, num_samples should not be specified, "
#                              "since a random permute will be performed.")

#         if not isinstance(self.num_samples, int) or self.num_samples <= 0:
#             raise ValueError("num_samples should be a positive integer "
#                              "value, but got num_samples={}".format(self.num_samples))

#     @property
#     def num_samples(self):
#         # dataset size might change at runtime
#         if self._num_samples is None:
#             return len(self.data_source)
#         return self._num_samples

#     def __iter__(self):
#         n = len(self.data_source)
#         if self.replacement:
#             return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
#         # print("-------------------------")
#         s = global_var.get_value('s')
#         return iter(s)

#     def __len__(self):
#         return self.num_samples


# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

class TrainTestSampler(Sampler):
    def __init__(self,data_source,is_training=True):
        super().__init__(data_source)
        self.data_source = data_source
        self.is_training = is_training
    def __iter__(self):
        if self.is_training:
            random.shuffle(self.data_source.res)
        return iter(self.data_source.res)

    def __len__(self):
        return len(self.data_source)
    
    
class VIDBatchSampler(BatchSampler):
    def __iter__(self):
        batch = []
        for ele in self.sampler:
            for filename in ele:
                batch.append(filename)
                if (len(batch)) == self.batch_size:
                    yield batch
                    batch = []
        if len(batch)>0 and not self.drop_last:
            yield batch
    def __len__(self):
        return len(self.sampler)


def create_dataloader_rgb_ir(path1, path2, imgsz, batch_size, stride, opt, temporal_stride, 
                                lframe, gframe, regex_search=r'.set...V...', hyp=None,
                                augment=False, cache=False, pad=0.0, rect=False, rank=-1, world_size=1,
                                workers=8, image_weights=False, quad=False, prefix='', sampler=None,
                                dataset_used='kaist', is_training=True, is_validation=False, 
                                temporal_mosaic=False, use_tadaconv=True,
                                sanitized=True, mosaic=False):
    
    with torch_distributed_zero_first(rank):
        dataset = LoadMultiModalImagesAndLabels(path1, path2, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=opt.single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix,
                                      temporal_stride=temporal_stride,
                                      lframe = lframe,
                                      gframe = gframe,
                                      regex_search=regex_search,
                                      dataset_used = dataset_used,
                                      is_training=is_training,
                                      temporal_mosaic=temporal_mosaic,
                                      use_tadaconv=use_tadaconv,
                                      sanitized=sanitized,
                                      mosaic=mosaic,
                                      )
    
    
    
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    
    if is_training and not is_validation and sampler == None:
        dataloader = loader(dataset,
                            batch_size=batch_size,
                            num_workers=nw,
                            sampler=sampler,
                            shuffle = True,
                            pin_memory=True,
                            drop_last=True, # Drop last to avoid error in multi-gpu
                            collate_fn=LoadMultiModalImagesAndLabels.collate_fn4 if quad else LoadMultiModalImagesAndLabels.collate_fn) 
        
    elif not is_training or is_validation:
        dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadMultiModalImagesAndLabels.collate_fn4 if quad else LoadMultiModalImagesAndLabels.collate_fn)
    else:
        raise 'Cant use sampler and shuffle'

    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride

        if pipe.isnumeric():
            pipe = eval(pipe)  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        print(f'webcam {self.count}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            if 'youtube.com/' in s or 'youtu.be/' in s:  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) % 100

            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f' success ({w}x{h} at {self.fps:.2f} FPS).')
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(1 / self.fps)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths, sanitized=False, dataset_used='kaist'):
    
    if dataset_used == 'CVC14':
        def format_path(path):
            pattern = r'/(\w+)/(\w+)/(\w+)/(\w+)/(\d+)\.png'
            
            match = re.search(pattern, path)
            if match:
                set_name = match.group(1)
                condition = match.group(2)
                seq = match.group(3)
                t = match.group(4)
                idx = match.group(5)
                return f"{t}/{set_name}_{condition}_{seq}_{idx}.txt"
            else:
                import pdb; pdb.set_trace()
                return None
            
        target = "/mnt/workspace/datasets/temp/CVC-14-Kaist-format/annotation/"
        res = [target + format_path(path) for path in img_paths if format_path(path) is not None]
        return res
    elif dataset_used == 'kaist':
        # Define label paths as a function of image paths
        if sanitized:
            
            def format_path(path):
                pattern = r'/images/(set\d+)/(\w+)/(\w+)/I(\d+)\.jpg'
                
                match = re.search(pattern, path)
                if match:
                    setID = match.group(1)
                    vid = match.group(2)
                    frameID = match.group(4)
                    return f"{setID}_{vid}_I{frameID}.txt"
                else:
                    return None

            # target = "/mnt/workspace/datasets/kaist-cvpr15/sanitized_annotations/sanitized_annotations_format/"
            target = KAIST_ANNOTATION_PATH
            res = [target + format_path(path) for path in img_paths if format_path(path) is not None]
            return res
        else:
            sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
            res = ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]
            return res
    elif dataset_used == 'camel':
        # TODO: Not sure if this is correct or not
        sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
        res = ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]
        return res
    else:
        raise 'Dataset not kaist, camel, or CVC14'

class LoadMultiModalImagesAndLabels(Dataset):  # for training/testing
    """
    FQY  载入多模态数据 （RGB 和 IR）
    """
    def __init__(self, path_rgb, path_ir, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='', regex_search = r'.set...V...', lframe = 16,
                 gframe=0, mode='gl', temporal_stride=6, dataset_used='kaist', is_training=True, delta = 0.0, temporal_mosaic=False, 
                 use_tadaconv=True, sanitized=True, mosaic=False):
        
        self.is_training = is_training
        self.temporal_stride = temporal_stride
        self.dataset_used = dataset_used
        self.batch_size = batch_size
        self.mode = mode
        self.lframe = lframe
        self.gframe = gframe
        self.regex_search = regex_search #need to change regex expression to match your dataset file path
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        # self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic = mosaic
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.delta = delta # relative pixel shift range +/- to images in sequences in temporal mosaic 
        self.temporal_mosaic = True if (self.augment and not self.rect ) and temporal_mosaic else False 
        self.stride = stride
        self.path_rgb = path_rgb
        self.path_ir = path_ir
        self.use_tadaconv = use_tadaconv
        self.sanitized = sanitized

        try:
            f_rgb = []  # image files
            f_ir = []
            # -----------------------------  rgb   -----------------------------
            for p_rgb in path_rgb if isinstance(path_rgb, list) else [path_rgb]:
                p_rgb = Path(p_rgb)  # os-agnostic
                if p_rgb.is_dir():  # dir
                    f_rgb += glob.glob(str(p_rgb / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p_rgb.is_file():  # file
                    with open(p_rgb, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p_rgb.parent) + os.sep
                        f_rgb += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{path_rgb} does not exist')

            # -----------------------------  ir   -----------------------------
            for p_ir in path_ir if isinstance(path_ir, list) else [path_ir]:
                p_ir = Path(p_ir)  # os-agnostic
                if p_ir.is_dir():  # dir
                    f_ir += glob.glob(str(p_ir / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p_ir.is_file():  # file
                    with open(p_ir, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p_ir.parent) + os.sep
                        f_ir += [x.replace('./', parent) if x.startswith('./') else x for x in
                                    t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p_ir} does not exist')

            self.img_files_rgb = sorted([x.replace('/', os.sep) for x in f_rgb if x.split('.')[-1].lower() in img_formats])
            self.img_files_ir = sorted([x.replace('/', os.sep) for x in f_ir if x.split('.')[-1].lower() in img_formats])

            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files_rgb, f'{prefix}No images found'
            assert self.img_files_ir, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path_rgb,path_ir}: {e}\nSee {help_url}')


        self.res = self.photo_to_sequence(lframe =self.lframe, 
                                          temporal_stride = self.temporal_stride,
                                          gframe = self.gframe)
        self.res_dict = {}
        for i, sequence in enumerate(self.res):
            self.res_dict[i] = sequence
        
        # This is so for training we only need to cache once 
        self.res_dict_ir = {}
        if not self.is_training:
            self.img_files_rgb = [frame for frames_tublet in self.res for frame in frames_tublet] # flattens nested list
            num_frames_in_seq = len(self.res[0])
            if dataset_used == 'kaist':
                self.img_files_ir = [frame.replace('visible', 'lwir') for frames_tublet in self.res for frame in frames_tublet] # flattens nested list                
            elif dataset_used == 'camel':
                self.img_files_ir = [frame.replace('camel', 'IR_camel') for frames_tublet in self.res for frame in frames_tublet] # flattens nested list
            elif dataset_used == 'CVC14':
                self.img_files_ir = [frame.replace('Visible', 'FIR') for frames_tublet in self.res for frame in frames_tublet] # flattens nested list
            else:
                assert 'Dataset not kaist, camel, or CVC14'
            # create path dict for thermal 
            l, r = 0, num_frames_in_seq
            index = 0
            while r <= len(self.img_files_ir):
                self.res_dict_ir[index] = self.img_files_ir[l:r]
                l = r
                r += num_frames_in_seq
                index += 1
            
            # Remove Duplicates in image paths
            self.img_files_rgb = sorted(list(set(self.img_files_rgb))) #removes Duplicates
            self.img_files_ir = sorted(list(set(self.img_files_ir))) #removes Duplicates
        else:
            for i, sequence in self.res_dict.items():
                if dataset_used == 'kaist':
                    self.res_dict_ir[i] = [frame.replace('visible', 'lwir') for frame in sequence]
                elif dataset_used == 'camel':
                    self.res_dict_ir[i] = [frame.replace('camel', 'IR_camel') for frame in sequence]
                elif dataset_used == 'CVC14':
                    self.res_dict_ir[i] = [frame.replace('Visible', 'FIR') for frame in sequence] # flattens nested list
                else:
                    assert 'Dataset not kaist, camel, or CVC14'
                    
        # Check cache
        # Check rgb cache
        self.label_files_rgb = img2label_paths(self.img_files_rgb, sanitized=self.sanitized, dataset_used=self.dataset_used)  # labels
        # print(self.label_files)
        cache_rgb_path = (p_rgb if p_rgb.is_file() else Path(self.label_files_rgb[0]).parent).with_suffix('.cache')  # cached labels
        if cache_rgb_path.is_file():
            cache_rgb, exists_rgb = torch.load(cache_rgb_path), True  # load
            if cache_rgb['hash'] != get_hash(self.label_files_rgb + self.img_files_rgb) or 'version' not in cache_rgb:  # changed
                print('Rebuilding Cache!!!!!!!!!!!!!')
                cache_rgb, exists_rgb = self.cache_labels(self.img_files_rgb, self.label_files_rgb,
                                                          cache_rgb_path, prefix), False  # re-cache
        else:
            cache_rgb, exists_rgb = self.cache_labels(self.img_files_rgb, self.label_files_rgb,
                                                      cache_rgb_path, prefix), False  # cache

        # Check ir cache
        self.label_files_ir = img2label_paths(self.img_files_ir, sanitized=self.sanitized, dataset_used=self.dataset_used)  # labels
        # print(self.label_files)
        cache_ir_path = (p_ir if p_ir.is_file() else Path(self.label_files_ir[0]).parent).with_suffix('.cache')  # cached labels
        if cache_ir_path.is_file():
            cache_ir, exists_ir = torch.load(cache_ir_path), True  # load
            if cache_ir['hash'] != get_hash(self.label_files_ir + self.img_files_ir) or 'version' not in cache_ir:  # changed
                cache_ir, exists_ir = self.cache_labels(self.img_files_ir, self.label_files_ir,
                                                        cache_ir_path, prefix), False  # re-cache
        else:
            cache_ir, exists_ir = self.cache_labels(self.img_files_ir, self.label_files_ir,
                                                    cache_ir_path, prefix), False  # cache
        # Display cache
        nf_rgb, nm_rgb, ne_rgb, nc_rgb, n_rgb = cache_rgb.pop('results')  # found, missing, empty, corrupted, total
        nf_ir, nm_ir, ne_ir, nc_ir, n_ir = cache_ir.pop('results')  # found, missing, empty, corrupted, total
        if exists_rgb:
            d = f"Scanning RGB '{cache_rgb_path}' images and labels... {nf_rgb} found, {nm_rgb} missing, {ne_rgb} empty, {nc_rgb} corrupted"
            tqdm(None, desc=prefix + d, total=n_rgb, initial=n_rgb)  # display cache results
        if exists_ir:
            d = f"Scanning IR '{cache_rgb_path}' images and labels... {nf_ir} found, {nm_ir} missing, {ne_ir} empty, {nc_ir} corrupted"
            tqdm(None, desc=prefix + d, total=n_ir, initial=n_ir)  # display cache results

        assert nf_rgb > 0 or not augment, f'{prefix}No labels in {cache_rgb_path}. Can not train without labels. See {help_url}'

        # Read cache
        # Read RGB cache
        cache_rgb.pop('hash')  # remove hash
        cache_rgb.pop('version')  # remove version
        labels_rgb, shapes_rgb, self.segments_rgb = zip(*cache_rgb.values())
        # self.labels_rgb = list(labels_rgb)
        self.img_files_rgb = list(cache_rgb.keys())  # update
        self.labels_rgb = {img_path: labels for img_path, labels in zip( self.img_files_rgb, labels_rgb)}
        
        
        self.shapes_rgb = np.array(shapes_rgb, dtype=np.float64)
        self.label_files_rgb = img2label_paths(cache_rgb.keys(), sanitized=self.sanitized, dataset_used=self.dataset_used)  # update
                
        if single_cls:
            for k in self.labels_rgb.keys():
                self.labels_rgb[k][:,0] = 0 

        n_rgb = len(shapes_rgb)  # number of images
        bi_rgb = np.floor(np.arange(n_rgb) / batch_size).astype(np.int_)  # batch index
        nb_rgb = bi_rgb[-1] + 1  # number of batches
        self.batch_rgb = bi_rgb  # batch index of image
        self.n_rgb = n_rgb
        self.indices_rgb = range(n_rgb)

        # Read IR cache
        cache_ir.pop('hash')  # remove hash
        cache_ir.pop('version')  # remove version
        labels_ir, shapes_ir, self.segments_ir = zip(*cache_ir.values())
        self.img_files_ir = list(cache_ir.keys())  # update
        self.labels_ir = {img_path: labels for img_path, labels in zip(self.img_files_ir, labels_ir)}        
        self.shapes_ir = np.array(shapes_ir, dtype=np.float64)
        self.label_files_ir = img2label_paths(cache_ir.keys(), sanitized=self.sanitized, dataset_used=self.dataset_used)  # update
        if single_cls:
            for k in self.labels_ir.keys():
                self.labels_ir[k][:,0] = 0


        n_ir = len(shapes_ir)  # number of images
        bi_ir = np.floor(np.arange(n_ir) / batch_size).astype(np.int_)  # batch index
        nb_ir = bi_ir[-1] + 1  # number of batches
        self.batch_ir = bi_ir  # batch index of image
        self.n_ir = n_ir
        self.indices_ir = range(n_ir)

        # print( "self.img_files_rgb,  self.img_files_ir")
        # print( self.img_files_rgb,  self.img_files_ir)

        # Rectangular Trainingx
        if self.rect:

            # RGB
            # Sort by aspect ratio
            s_rgb = self.shapes_rgb  # wh
            ar_rgb = s_rgb[:, 1] / s_rgb[:, 0]  # aspect ratio
            # irect_rgb = ar_rgb.argsort() #breaks ordered video structure 
            # self.img_files_rgb = [self.img_files_rgb[i] for i in irect_rgb]
            # self.label_files_rgb = [self.label_files_rgb[i] for i in irect_rgb]
            # self.labels_rgb = [self.labels_rgb[i] for i in irect_rgb]
            # self.shapes_rgb = s_rgb[irect_rgb]  # wh
            # ar_rgb = ar_rgb[irect_rgb]

            # Set training image shapes
            shapes_rgb = [[1, 1]] * nb_rgb
            for i in range(nb_rgb):
                ari_rgb = ar_rgb[bi_rgb == i]
                mini, maxi = ari_rgb.min(), ari_rgb.max()
                if maxi < 1:
                    shapes_rgb[i] = [maxi, 1]
                elif mini > 1:
                    shapes_rgb[i] = [1, 1 / mini]

            self.batch_shapes_rgb = np.ceil(np.array(shapes_rgb) * img_size / stride + pad).astype(np.int_) * stride

            # IR
            # Sort by aspect ratio
            s_ir = self.shapes_ir  # wh
            ar_ir = s_ir[:, 1] / s_ir[:, 0]  # aspect ratio
            # irect_ir = ar_ir.argsort()
            # self.img_files_ir = [self.img_files_ir[i] for i in irect_ir]
            # self.label_files_ir = [self.label_files_ir[i] for i in irect_ir]
            # self.labels_ir = [self.labels_ir[i] for i in irect_ir]
            # self.shapes_ir = s_ir[irect_ir]  # wh
            # ar_ir = ar_ir[irect_ir]

            # Set training image shapes
            shapes_ir = [[1, 1]] * nb_ir
            for i in range(nb_ir):
                ari_ir = ar_ir[bi_ir == i]
                mini, maxi = ari_ir.min(), ari_ir.max()
                if maxi < 1:
                    shapes_ir[i] = [maxi, 1]
                elif mini > 1:
                    shapes_ir[i] = [1, 1 / mini]

            self.batch_shapes_ir = np.ceil(np.array(shapes_ir) * img_size / stride + pad).astype(np.int_) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        # self.imgs_rgb = [None] * n_rgb
        # self.imgs_ir = [None] * n_ir

        self.labels = self.labels_rgb
        self.shapes = self.shapes_rgb
        self.indices = self.indices_rgb
        
        


    def cache_labels(self, imgfiles, labelfiles, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        img_files = imgfiles
        label_files = labelfiles
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(img_files, label_files), desc='Scanning images', total=len(img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im.format.lower() in img_formats, f'invalid image format {im.format}'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines()]
                        if any([len(x) > 8 for x in l]):  # is segment
                            classes = np.array([x[0] for x in l], dtype=np.float32)
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                            l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                        l = np.array(l, dtype=np.float32)
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape, segments]
            except Exception as e:
                nc += 1
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(label_files + img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.1  # cache version
        torch.save(x, path)  # save for next time
        logging.info(f'{prefix}New cache created: {path}')
        return x

    def photo_to_sequence(self, lframe, temporal_stride, gframe=0):
        """"
        Args:
            dataset_paths: self.img_files_rgb
        
        """
        if self.dataset_used == 'kaist':
            vid = re.search(self.regex_search, self.img_files_rgb[0])
            vid_old = vid.string[vid.regs[0][0]:vid.regs[0][1]]
        elif self.dataset_used == 'CVC14':
            pattern = r'/(\w+)/(\w+)/(\w+)/(\w+)/(\d+)\.png'
            print(self.img_files_rgb[0])
            match = re.search(pattern, self.img_files_rgb[0])
            vid_old = match.group(5)
        sorted_videos = [] # will end up as a list of lists
        one_vid = []
        
        #  Need to separate the entire dataset into videos format [[video 1],[video 2],...]
        for index, img_path in  enumerate(self.img_files_rgb):
            if self.dataset_used == 'kaist':
                vid = re.search(self.regex_search, img_path)
                vid_new = vid.string[vid.regs[0][0]:vid.regs[0][1]]
            elif self.dataset_used == 'CVC14':
                match = re.search(pattern, img_path)
                vid_new = match.group(5)
            if vid_new != vid_old:
                sorted_videos.append(one_vid)
                one_vid = []
                one_vid.append(img_path)
                vid_old = vid_new
            else:
                one_vid.append(img_path)
                if index == len(self.img_files_rgb) - 1:
                    sorted_videos.append(one_vid)
        
        res = []
        lwindow = lframe*temporal_stride - temporal_stride + 1
        if lwindow != 1:
            self.num_of_images_in_one_sequence = len(np.arange(0,lwindow, temporal_stride))
        else:
            self.num_of_images_in_one_sequence = 1

        # do not create gwindow because gframe can be chosen from any frame/image in a specfic video
        
        for video in sorted_videos:
            ele_len = len(video)
            if self.is_training:  
                # if not self.sanitized:  
                #     if ele_len<lwindow+gframe:
                #         #TODO fix the unsolved part
                #         #res.append(video)
                #         continue
                #     else:
                #         if self.mode == 'random':
                #             split_num = int(ele_len / (gframe))
                #             random.shuffle(video)
                #             for i in range(split_num):
                #                 res.append(video[i * gframe:(i + 1) * gframe])
                #         elif self.mode == 'uniform':
                #             split_num = int(ele_len / (gframe))
                #             all_uniform_frame = video[:split_num * gframe]
                #             for i in range(split_num):
                #                 res.append(all_uniform_frame[i::split_num])
                #         elif self.mode == 'gl':
                #             split_num = int(ele_len / (lwindow))
                #             all_local_frame = video[:split_num * lwindow]
                #             for i in range(split_num):
                #                 g_frame = random.sample(video[:i * lwindow] + video[(i + 1) * lwindow:], gframe)
                #                 # res.append(all_local_frame[i * lwindow:(i + 1) * lwindow] + g_frame)
                #                 res.append(all_local_frame[i * lwindow:(i + 1) * lwindow:temporal_stride] + g_frame)
                #         else:
                #             print('unsupport mode, exit')
                #             exit(0)
                # else:
                if self.dataset_used in ['kaist', 'CVC14']:
                    # Need to think about the inital begining part here 
                    if self.mode == 'gl':
                        # pseduo_end_frame = int(video[-1][-9:-4])
                        for frame in video:
                            if gframe == 0:
                                g_frame = []
                            else:
                                raise 'gframe not zero'
                            nearby_strided_frames = self.get_frames_tublet(frame, lwindow, temporal_stride)
                            if nearby_strided_frames is not None:
                                res.append(nearby_strided_frames + g_frame)
                else:
                    raise 'Not the kaist or CVC14 dataset'
               
                
            else:
                # Assuming Kaist input data is spare test subset (test20all.txt)
                # meaning that ele_len < lwindow + gframe might not always be true for spare test subset
                if self.dataset_used in ['kaist', 'CVC14']:
                    # Need to think about the inital begining part here 
                    if self.mode == 'gl':
                        # pseduo_end_frame = int(video[-1][-9:-4])
                        for frame in video:
                            if gframe == 0:
                                g_frame = []
                            else:
                                raise 'gframe not zero'
                            nearby_strided_frames = self.get_frames_tublet(frame, lwindow, temporal_stride)
                            res.append(nearby_strided_frames + g_frame)
                else:
                    raise 'Not the kaist or CVC14 dataset'
                    
        return res  
          
    def get_frames_tublet(self, frame, lwindow, temporal_stride):
        """
        This is currently designed only for testing on Kaist
        Because common Kaist Eval is on spare subset
        Takses in a String, and returns a list of strings
        """
        new_frames = []
        if self.dataset_used in ['kaist', 'CVC14']:
            frame_num = int(frame[-9:-4])
            start_idx = frame_num+1-lwindow
            end_idx = frame_num+1
            if start_idx < 0:
                raise 'Have not implemented test load edge cases'
            
            frames_to_add = np.arange(start_idx, end_idx, temporal_stride)
            for _frame in frames_to_add:
                if self.dataset_used == 'kaist':
                    frame_path = os.path.join( frame[:-10] , 'I{:05d}'.format(_frame) + '.jpg')
                elif self.dataset_used == 'CVC14':
                    frame_path = os.path.join( frame[:-9] , '{:05d}'.format(_frame) + '.png')
                new_frames.append(frame_path)
            
        else:
            raise 'Not the kaist or CVC14 dataset'
        
        return new_frames
        
        
    def __len__(self):
        return len(self.res)
        # return len(self.img_files_rgb)

    def __getitem__(self, index):
        # Load image
        hyp = self.hyp
        #this is saying that we input a sequence of images for video based multimodal fusion
        # however we only use the last image in the sequence for the labels
        index_to_look_at = -1
        
        if self.mosaic:
            if random.random() <= 0.5:
                if self.temporal_mosaic:
                    # below loads temporal mosaic 
                    imgs_rgb, labels_rgb, imgs_ir, labels_ir = load_mosaic_RGB_IR_temporal(self, index, 
                                                                                        self.delta, self.temporal_mosaic)
                else:
                    # below loads moasic images are random
                    imgs_rgb, labels_rgb, imgs_ir, labels_ir = load_mosaic_RGB_IR_temporal(self, index, 
                                                                                        self.delta, self.temporal_mosaic)
                shapes = None
            else:
                imgs_rgb, imgs_ir, (h0, w0), (h, w) = load_image_rgb_ir_temporal(self, index)
                shape = self.batch_shapes_rgb[0] if self.rect else self.img_size  # final letterboxed shape
                

                imgs_rgb, ratio, pad = letterbox_temporal(imgs_rgb, shape, auto=False, scaleup=self.augment)
                imgs_ir, ratio, pad = letterbox_temporal(imgs_ir, shape, auto=False, scaleup=self.augment)
                            
                
                shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        
                rgb_path = self.res_dict[index][index_to_look_at]
                if rgb_path in self.labels_rgb.keys():
                    labels_rgb = self.labels_rgb[rgb_path].copy()
                else:
                    labels_rgb = np.zeros((0, 5), dtype=np.float32)
                    print("Warning: No labels for RGB image: Due to non-normalized or out of bounds coordinate labels")
                if labels_rgb.size:  # normalized xywh to pixel xyxy format
                    labels_rgb[:, 1:] = xywhn2xyxy(labels_rgb[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                ir_path = self.res_dict_ir[index][index_to_look_at]
                if ir_path in self.labels_ir.keys():
                    labels_ir = self.labels_ir[ir_path].copy()
                else:
                    labels_ir = np.zeros((0, 5), dtype=np.float32)
                    print("Warning: No labels for IR image: Due to non-normalized or out of bounds coordinate labels")
                if labels_ir.size:  # normalized xywh to pixel xyxy format
                    labels_ir[:, 1:] = xywhn2xyxy(labels_ir[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
               
                
                
        else:
        # Letterbox
            imgs_rgb, imgs_ir, (h0, w0), (h, w) = load_image_rgb_ir_temporal(self, index)
            # img_rgb, img_ir, (h0, w0), (h, w) = load_image_rgb_ir(self, path, self.dataset_used)
            
            # since the batch_shape size is the same in Kaist and Camel fixed it
            shape = self.batch_shapes_rgb[0] if self.rect else self.img_size  # final letterboxed shape
            

            imgs_rgb, ratio, pad = letterbox_temporal(imgs_rgb, shape, auto=False, scaleup=self.augment)
            imgs_ir, ratio, pad = letterbox_temporal(imgs_ir, shape, auto=False, scaleup=self.augment)
            
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            rgb_path = self.res_dict[index][index_to_look_at]
            if rgb_path in self.labels_rgb.keys():
                labels_rgb = self.labels_rgb[rgb_path].copy()
            else:
                labels_rgb = np.zeros((0, 5), dtype=np.float32)
                print("Warning: No labels for RGB image: Due to non-normalized or out of bounds coordinate labels")
            if labels_rgb.size:  # normalized xywh to pixel xyxy format
                labels_rgb[:, 1:] = xywhn2xyxy(labels_rgb[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
            ir_path = self.res_dict_ir[index][index_to_look_at]
            if ir_path in self.labels_ir.keys():
                labels_ir = self.labels_ir[ir_path].copy()
            else:
                labels_ir = np.zeros((0, 5), dtype=np.float32)
                print("Warning: No labels for IR image: Due to non-normalized or out of bounds coordinate labels")
            if labels_ir.size:  # normalized xywh to pixel xyxy format
                labels_ir[:, 1:] = xywhn2xyxy(labels_ir[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            
        path = self.res_dict[index][index_to_look_at]
        if self.augment:
            # Augment colorspace
            imgs_rgb = augment_hsv_temporal(imgs_rgb, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v']) #not in_place
            imgs_ir = augment_hsv_temporal(imgs_ir, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
        
        nL_rgb = len(labels_rgb)  # number of labels
        if nL_rgb:
            labels_rgb[:, 1:5] = xyxy2xywh(labels_rgb[:, 1:5])  # convert xyxy to xywh
            labels_rgb[:, [2, 4]] /= imgs_rgb.shape[0]  # normalized height 0-1
            labels_rgb[:, [1, 3]] /= imgs_rgb.shape[1]  # normalized width 0-1
        nL_ir = len(labels_ir)  # number of labels
        if nL_ir:
            labels_ir[:, 1:5] = xyxy2xywh(labels_ir[:, 1:5])  # convert xyxy to xywh
            labels_ir[:, [2, 4]] /= imgs_ir.shape[0]  # normalized height 0-1
            labels_ir[:, [1, 3]] /= imgs_ir.shape[1]  # normalized width 0-1

        
        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                imgs_rgb = np.flipud(imgs_rgb)
                imgs_ir = np.flipud(imgs_ir)
                if nL_rgb: labels_rgb[:, 2] = 1 - labels_rgb[:, 2]
                if nL_ir: labels_ir[:, 2] = 1 - labels_ir[:, 2]

            # flip left-right
            if random.random() < hyp['fliplr']:
                imgs_rgb = np.fliplr(imgs_rgb)
                imgs_ir = np.fliplr(imgs_ir)

                if nL_rgb: labels_rgb[:, 1] = 1 - labels_rgb[:, 1]
                if nL_ir: labels_ir[:, 1] = 1 - labels_ir[:, 1]

        labels_out_rgb = torch.zeros((nL_rgb, 6))
        if nL_rgb: labels_out_rgb[:, 1:] = torch.from_numpy(labels_rgb)
        
        labels_out_ir = torch.zeros((nL_ir, 6))
        if nL_ir: labels_out_ir[:, 1:] = torch.from_numpy(labels_ir)
            
        if self.use_tadaconv:
            imgs_rgb = imgs_rgb[:, :, ::-1].reshape(imgs_rgb.shape[0], imgs_rgb.shape[1], self.lframe, 3).transpose(3,2,0,1)
            imgs_ir = imgs_ir[:, :, ::-1].reshape(imgs_ir.shape[0], imgs_ir.shape[1], self.lframe, 3).transpose(3,2,0,1)    
        else:
            imgs_rgb = imgs_rgb[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            imgs_ir = imgs_ir[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416

        imgs_rgb = np.ascontiguousarray(imgs_rgb)
        imgs_ir = np.ascontiguousarray(imgs_ir)
        
        imgs_all = np.concatenate((imgs_rgb, imgs_ir), axis=0)

        return torch.from_numpy(imgs_all), labels_out_rgb, labels_out_ir, path, shapes
        

    @staticmethod
    def collate_fn_temporal(batch):
        imgs_rgb, imgs_ir, labels, paths, shapes, num_of_images_in_one_sequence = zip(*batch)  # transposed
        # num_of_images_in_one_sequence tells the number of images that are images 
        # for i, l in enumerate(labels):
        #     l[:, 0] = i  # add target image index for build_targets()
        
        num_images_one_seq = num_of_images_in_one_sequence[0]
        num_batches = len(imgs_rgb)//num_images_one_seq
        
        #this is saying that we input a sequence of images for video based multimodal fusion
        # however we only use the last image in the sequence for the labels
        index_to_look_at = -1
        


        
        paths_resized = [paths[i*num_images_one_seq:(i+1)*num_images_one_seq][index_to_look_at] for i in range(num_batches)]
        shapes_resized = [shapes[i*num_images_one_seq:(i+1)*num_images_one_seq][index_to_look_at] for i in range(num_batches)]
        
        labels_resized = [labels[i*num_images_one_seq:(i+1)*num_images_one_seq][index_to_look_at] for i in range(num_batches)]
        
        for i, l in enumerate(labels_resized):
            l[:, 0] = i  # add target image index for build_targets()

        imgs_rgb = torch.stack(imgs_rgb, 0)
        h,w = imgs_rgb.shape[-2], imgs_rgb.shape[-1]
        imgs_rgb = imgs_rgb.reshape(-1, 3*num_images_one_seq, h, w)
        
        imgs_ir = torch.stack(imgs_ir, 0)
        imgs_ir = imgs_ir.reshape(-1,3*num_images_one_seq, h, w)
        
        # return torch.cat((imgs_rgb, imgs_ir), 1), torch.cat(labels, 0), paths, shapes
        
        return torch.cat((imgs_rgb, imgs_ir), 1), torch.cat(labels_resized, 0), tuple(paths_resized), tuple(shapes_resized)
    
    @staticmethod
    def collate_fn(batch):
        img, label_rgb, label_ir, path, shapes = zip(*batch)  # transposed
        for i, l_rgb in enumerate(label_rgb):
            l_rgb[:, 0] = i  # add target image index for build_targets()
        for i, l_ir in enumerate(label_ir):
            l_ir[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label_rgb, 0), torch.cat(label_ir, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        label_rgb, label_ir = label[0], label[1]
        n = len(shapes) // 4
        # img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]
        img4, label4_rgb, label4_ir, path4, shapes4 = [], [], [], path, shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l_rgb = label_rgb[i]
                l_ir = label_ir[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l_rgb = torch.cat((label_rgb[i], label_rgb[i + 1] + ho, label_rgb[i + 2] + wo, label_rgb[i + 3] + ho + wo), 0) * s
                l_ir = torch.cat((label_ir[i], label_ir[i + 1] + ho, label_ir[i + 2] + wo, label_ir[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4_rgb.append(l_rgb)
            label4_ir.append(l_ir)

        for i, l_rgb in enumerate(label4_rgb):
            l_rgb[:, 0] = i  # add target image index for build_targets()
        for i, l_ir in enumerate(label4_ir):
            l_ir[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), (torch.cat(label4_rgb, 0), torch.cat(label4_ir, 0)), path4, shapes4



# Ancillary functions --------------------------------------------------------------------------------------------------

def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def load_image_rgb_ir_temporal(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    if type(index) == int or type(index) == np.int64:
        paths_rgb = self.res_dict[index]
        paths_ir = self.res_dict_ir[index] # rgb and ir are on same index 
    elif type(index) == list:
        paths_rgb, paths_ir = [], []
        for i in index:
            paths_rgb.append(self.img_files_rgb[i])
            paths_ir.append(self.img_files_ir[i])
    else:
        raise 'Error index not int/np.int64 or list'

    imgs_rgb = []
    imgs_ir = []
    for path_rgb, path_ir in zip(paths_rgb, paths_ir):
        img_rgb = cv2.imread(path_rgb)
        img_ir = cv2.imread(path_ir)
        
        if img_rgb is None:
            print('Image RGB Not Found ' + path_rgb)
        if img_ir is None:
            print('Image IR Not Found ' + path_ir)
        
        assert img_rgb is not None, 'Image RGB Not Found ' + path_rgb
        assert img_ir is not None, 'Image IR Not Found ' + path_ir

        
        h0, w0 = img_rgb.shape[:2]  # orig hw, shouldnt change for a specfic sequence
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img_rgb = cv2.resize(img_rgb, (int(w0 * r), int(h0 * r)),
                                interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
            img_ir = cv2.resize(img_ir, (int(w0 * r), int(h0 * r)),
                                interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
          
        imgs_rgb.append(img_rgb)
        imgs_ir.append(img_ir)
        
        
    imgs_rgb = np.concatenate(imgs_rgb, axis=-1) #channel is last dim
    imgs_ir = np.concatenate(imgs_ir, axis=-1) #channel is last dim
    
    return imgs_rgb, imgs_ir, (h0, w0), imgs_rgb.shape[:2]  # img, hw_original, hw_resized



def load_image_rgb_ir(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw

    path_rgb = self.img_files_rgb[index]
    path_ir = self.img_files_ir[index]

    img_rgb = cv2.imread(path_rgb)  # BGR
    # might need to change path_ir above replacement
    img_ir = cv2.imread(path_ir)  # BGR
        
    

    assert img_rgb is not None, 'Image RGB Not Found ' + path_rgb
    assert img_ir is not None, 'Image IR Not Found ' + path_ir

    h0, w0 = img_rgb.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        img_rgb = cv2.resize(img_rgb, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        img_ir = cv2.resize(img_ir, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
    return img_rgb, img_ir, (h0, w0), img_rgb.shape[:2]  # img, hw_original, hw_resized

def augment_hsv_temporal(imgs, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    dtype = imgs.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    three_channel_indices = np.arange(0, imgs.shape[2]/3 + 1, dtype=int)*3 #assume that both RGB and thermal images are three channel
    l = three_channel_indices[0]
    
    imgs_list = []
    for r in three_channel_indices[1:]:
        hue, sat, val = cv2.split(cv2.cvtColor(imgs[:,:,l:r], cv2.COLOR_BGR2HSV))
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # cant do inplace ouput
        imgs_list.append(img)
        l = r 
    
    return np.concatenate(imgs_list, axis=-1)

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def hist_equalize(img, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'img' with img.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB

def load_mosaic_RGB_IR_temporal(self, index, delta, temporal_mosaic=True):
    """
    self: dataset object
    index: current index
    delta: value to shift images by 
    temporal_mosaic: load sequences or images
    """
    # loads images in a 4-mosaic
    index_rgb = index
    index_ir = index
    
    
    
    # number of local frames to load
    # lframes = self.res_dict[index]
    labels4_rgb, segments4_rgb = [], []
    labels4_ir, segments4_ir = [], []
    # labels4_rgb_list, segments4_rgb_list = [[] for _ in range(lframes)], [[] for _ in range(lframes)]
    # labels4_ir_list, segments4_ir_list = [[] for _ in range(lframes)], [[] for _ in range(lframes)]
    

    s = self.img_size


    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y

    assert index_rgb == index_ir, 'INDEX RGB 不等于 INDEX IR'

    if temporal_mosaic:
        indices = [index_rgb] + random.choices(np.arange(0,len(self.res)), k=3)  # 3 additional image indices
    else:
        num_frames_in_seq = len(self.res[0])
        rand = random.choices(self.indices_rgb, k=3*num_frames_in_seq)
        random_indices =  [rand[i:i+num_frames_in_seq] for i in range(0,3*num_frames_in_seq, num_frames_in_seq)]       
        indices = [index_rgb] + random_indices  # 3 additional image indices

    for i, index in enumerate(indices):

        # img, _, (h, w) = load_image(self, index)
        imgs_rgb, imgs_ir, _, (h, w) = load_image_rgb_ir_temporal(self, index)
        # cv2.imwrite("rgb_%s.jpg"%str(index), img_rgb)
        # cv2.imwrite("ir_%s.jpg"%str(index), img_ir)

        # place img in img4
        if i == 0:  # top left
            img4_rgb = np.full((s * 2, s * 2, imgs_rgb.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            img4_ir = np.full((s * 2, s * 2, imgs_ir.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        # padw = x1a - x1b
        # padh = y1a - y1b

        img4_rgb[y1a:y2a, x1a:x2a] = imgs_rgb[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        img4_ir[y1a:y2a, x1a:x2a] = imgs_ir[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b
    
        # Labels
        # This is saying that we input a sequence of images for video based multimodal fusion
        # however only  the labels of last image in the sequence are used
        index_to_look_at = -1
        
        if temporal_mosaic or i == 0: #when i=0 temporal sequence
            segments_rgb = self.segments_rgb[index].copy() #empty
            rgb_path = self.res_dict[index][index_to_look_at]
            if rgb_path in self.labels_rgb.keys():
                labels_rgb = self.labels_rgb[rgb_path].copy()
            else:
                labels_rgb = np.zeros((0, 5), dtype=np.float32)
                print("Warning: No labels for RGB image: Due to non-normalized or out of bounds coordinate labels")
            segments_ir = self.segments_ir[index].copy()
            ir_path = self.res_dict_ir[index][index_to_look_at]
            if ir_path in self.labels_ir.keys():
                labels_ir = self.labels_ir[ir_path].copy()
            else:
                labels_ir = np.zeros((0, 5), dtype=np.float32)
                print("Warning: No labels for IR image: Due to non-normalized or out of bounds coordinate labels")
            # segments_rgb = self.segments_rgb[index].copy() #empty
            # labels_rgb = [self.labels_rgb[i].copy() for i in self.res_dict[index]][index_to_look_at] # i here is path

            # segments_ir = self.segments_ir[index].copy()
            # labels_ir = [self.labels_ir[i].copy() for i in self.res_dict_ir[index]][index_to_look_at] # i here is path
        else:
            segments_rgb = self.segments_rgb[index[0]].copy() #empty
            rgb_path = self.img_files_rgb[i]
            if rgb_path in self.labels_rgb.keys():
                labels_rgb = [self.labels_rgb[rgb_path].copy() for i in index][index_to_look_at] # i here in int
            else:
                labels_rgb = np.zeros((0, 5), dtype=np.float32)
                print("Warning: No labels for RGB image: Due to non-normalized or out of bounds coordinate labels")

            segments_ir = self.segments_ir[index[0]].copy()
            ir_path = self.img_files_ir[i]
            if ir_path in self.labels_ir.keys():
                labels_ir = [self.labels_ir[ir_path].copy() for i in index][index_to_look_at]
            else:
                labels_ir = np.zeros((0, 5), dtype=np.float32)
                print("Warning: No labels for IR image: Due to non-normalized or out of bounds coordinate labels")

        if labels_rgb.size:
            labels_rgb[:, 1:] = xywhn2xyxy(labels_rgb[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            labels_ir[:, 1:] = xywhn2xyxy(labels_ir[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments_rgb = [xyn2xy(x, w, h, padw, padh) for x in segments_rgb]
            segments_ir = [xyn2xy(x, w, h, padw, padh) for x in segments_ir]
        labels4_rgb.append(labels_rgb)
        segments4_rgb.extend(segments_rgb)
        labels4_ir.append(labels_ir)
        segments4_ir.extend(segments_ir)
        

    # # Concat/clip labels

    start = 1
    labels4_rgb = np.concatenate(labels4_rgb, 0)
    labels4_ir = np.concatenate(labels4_ir, 0)
    for x in (labels4_rgb[:, start:], *segments4_rgb):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    for x in (labels4_ir[:, start:], *segments4_ir):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()

    # # Augment

    img4_rgb_new, img4_ir_new, labels4_rgb_new, labels4_ir_new = \
        random_perspective_rgb_ir_temporal(img4_rgb, 
                                           img4_ir, 
                                           labels4_rgb, 
                                           labels4_ir,
                                           segments4_rgb, 
                                           segments4_ir,
                                            degrees=self.hyp['degrees'],
                                            translate=self.hyp['translate'],
                                            scale=self.hyp['scale'],
                                            shear=self.hyp['shear'],
                                            perspective=self.hyp['perspective'],
                                            border=self.mosaic_border,
                                            delta = delta)  # border to remove

    if self.dataset_used == 'kaist':
        labels4_ir_new = labels4_rgb_new.copy()   

    return img4_rgb_new, labels4_rgb_new, img4_ir_new, labels4_ir_new





def letterbox_temporal(imgs, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = imgs.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        imgs = cv2.resize(imgs, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    imgs_list = []
    three_channel_indices = np.arange(0, imgs.shape[2]/3 + 1, dtype=int)*3 #assume that both RGB and thermal images are three channel
    l = three_channel_indices[0]
    for r in three_channel_indices[1:]:
        img = cv2.copyMakeBorder(imgs[:,:,l:r], top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        imgs_list.append(img)
        l = r
    

    imgs = np.concatenate(imgs_list, axis=-1)
    
    return imgs, ratio, (dw, dh)



def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def label_transform(targets, segments, M, s, width, height, perspective):
    # Transform label coordinates
    n = len(targets)
    targets_shape = targets.shape[1] #should be 6 for fullframes, 5 for lastframe /default
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            if targets_shape == 6:
                xy[:, :2] = targets[:, [2, 3, 4, 5, 2, 5, 4, 3]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            elif targets_shape == 5: 
                xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            else:
                assert targets_shape in {5, 6}, 'labels require 5 or 6 columns each'
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        end_i = 6 if targets_shape == 6 else 5
        start_i = 2 if targets_shape == 6 else 1
        # filter candidates
        i = box_candidates(box1=targets[:, start_i:end_i].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, start_i:end_i] = new[i]
    return targets

def random_perspective_rgb_ir_temporal(imgs_rgb, imgs_ir, targets_rgb=(),targets_ir=(), segments_rgb=(), segments_ir=(),
                              degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0), delta=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    # img = imgs_rgb
    # targets = targets_rgb
    # segments = segments_rgb
    assert imgs_rgb.shape[0] == imgs_ir.shape[0], f'imgs_rgb.shape[0] ({imgs_rgb.shape[0]}) != imgs_ir.shape[0] ({imgs_ir.shape[0]})'

    height = imgs_rgb.shape[0] + border[0] * 2  # shape(h,w,c)
    width = imgs_rgb.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -imgs_rgb.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -imgs_rgb.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M_orginal = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    
    offset_gt = []
    cropped_imgs_rgb = []
    cropped_imgs_ir = []
    three_channel_indices = np.arange(0, imgs_rgb.shape[2]/3 + 1, dtype=int)*3 #assuming that both RGB and thermal images are three channel 
    l = three_channel_indices[0]
    for index, r in enumerate(three_channel_indices[1:]):
        O = np.eye(3)

        if index != len(three_channel_indices) - 2:
            O[0, 2] = random.uniform(-delta, delta) * width
            O[1, 2] = random.uniform(-delta, delta) * height
            offset_gt.append([O[0,2] , O[1,2]]) #[x_offset, y_offset]
        else:
            offset_gt.append([0,0]) #[x_offset, y_offset]

        M = O @ M_orginal 
        
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                # img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
                img_rgb = cv2.warpPerspective(imgs_rgb[:,:,l:r], M, dsize=(width, height), borderValue=(114, 114, 114))
                img_ir = cv2.warpPerspective(imgs_ir[:,:,l:r], M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                # img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
                img_rgb = cv2.warpAffine(imgs_rgb[:,:,l:r], M[:2], dsize=(width, height), borderValue=(114, 114, 114))
                img_ir = cv2.warpAffine(imgs_ir[:,:,l:r], M[:2], dsize=(width, height), borderValue=(114, 114, 114))
                
            cropped_imgs_rgb.append(img_rgb)
            cropped_imgs_ir.append(img_ir)
            l = r

    imgs_rgb = np.concatenate(cropped_imgs_rgb, axis=-1)
    imgs_ir = np.concatenate(cropped_imgs_ir, axis=-1)
    
    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img_rgb[:, :, ::-1])  # base
    # ax[1].imshow(img_ir[:, :, ::-1])  # warped


    target_rgb = label_transform(targets_rgb, segments_rgb, M, s, width, height, perspective)
    target_ir = label_transform(targets_ir, segments_ir, M, s, width, height, perspective)
    
    return imgs_rgb, imgs_ir, target_rgb, target_ir



def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates
