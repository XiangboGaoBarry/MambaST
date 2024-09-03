# YOLOv5 PyTorch utils

import datetime
import logging
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None
logger = logging.getLogger(__name__)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def git_describe(path=Path(__file__).parent):  # path must be a directory
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''  # not a git repository


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'YOLOv5 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def profile(x, ops, n=100, device=None):
    # profile a pytorch module or list of modules. Example usage:
    #     x = torch.randn(16, 3, 640, 640)  # input
    #     m1 = lambda x: x * torch.sigmoid(x)
    #     m2 = nn.SiLU()
    #     profile(x, [m1, m2], n=100)  # profile speed over 100 iterations

    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    x.requires_grad = True
    print(torch.__version__, device.type, torch.cuda.get_device_properties(0) if device.type == 'cuda' else '')
    print(f"\n{'Params':>12s}{'GFLOPS':>12s}{'forward (ms)':>16s}{'backward (ms)':>16s}{'input':>24s}{'output':>24s}")
    for m in ops if isinstance(ops, list) else [ops]:
        m = m.to(device) if hasattr(m, 'to') else m  # device
        m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m  # type
        dtf, dtb, t = 0., 0., [0., 0., 0.]  # dt forward, backward
        try:
            flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # GFLOPS
        except:
            flops = 0

        for _ in range(n):
            t[0] = time_synchronized()
            y = m(x)
            t[1] = time_synchronized()
            try:
                _ = y.sum().backward()
                t[2] = time_synchronized()
            except:  # no backward method
                t[2] = float('nan')
            dtf += (t[1] - t[0]) * 1000 / n  # ms per op forward
            dtb += (t[2] - t[1]) * 1000 / n  # ms per op backward

        s_in = tuple(x.shape) if isinstance(x, torch.Tensor) else 'list'
        s_out = tuple(y.shape) if isinstance(y, torch.Tensor) else 'list'
        p = sum(list(x.numel() for x in m.parameters())) if isinstance(m, nn.Module) else 0  # parameters
        print(f'{p:12}{flops:12.4g}{dtf:16.4g}{dtb:16.4g}{str(s_in):>24s}{str(s_out):>24s}')


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def save_intersect_dicts_tadaconv(tgt, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    # save = [f'model.{i}.' for i in range(0,5)]
    # save += [f'model.{9}.']
    layers_to_change = 'transferred.txt'
    # save_set = set(save)  
    for k, v in tgt.items():
        # if k[:8] not in save_set:
        #     continue
        
        # if 'rf_func' in k:
        #     continue
        # if '.bn_b.' in k:
        #     continue
        with open(layers_to_change, 'a') as f:
            f.write(f'{k}' + '\n' )



def intersect_dicts_full(src, tgt, mode=None, back_or_head='backbone', tadaconv=True, baexclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    # skip = [f'model.{i}.' for i in range(10,47)]
    # skip += [f'model.{i}.' for i in [5,6,7,8]]
    backbone = [f'model.{i}.' for i in range(0,10)]
    head = [f'model.{i}.' for i in range(10, 24)]
    if back_or_head == 'backbone':
        if mode == 'rgb': 
            tada_conv_yolov5l_backbone_num = [str(model_num) for model_num in range(0,5)]
            tada_conv_yolov5l_backbone_num += [str(model_num) for model_num in range(13,15)]
            tada_conv_yolov5l_backbone_num += [str(model_num) for model_num in range(20,23)]

        elif mode == 'ir':
            tada_conv_yolov5l_backbone_num = [str(model_num) for model_num in range(5,10)]
            tada_conv_yolov5l_backbone_num += [str(model_num) for model_num in range(15,17)]
            tada_conv_yolov5l_backbone_num += [str(model_num) for model_num in range(23,26)]

        yolov5l_backbone_num =  [str(i) for i in range(0,10)]
        backbone_map = dict(zip(yolov5l_backbone_num, tada_conv_yolov5l_backbone_num)) #yolov5l -> tadaconv backbone mapping
        check_model = backbone
    elif back_or_head == 'head':
        tada_conv_yolov5l_head_num = [str(model_num) for model_num in range(32,46)]
        yolov5l_head_num =  [str(i) for i in range(10,24)]
        head_map = dict(zip(yolov5l_head_num, tada_conv_yolov5l_head_num))
        check_model = head
    elif back_or_head == 'headRGB':
        tada_conv_yolov5l_head_num = [str(model_num) for model_num in range(32,46)]
        yolov5l_head_num =  [str(i) for i in range(10,24)]
        head_map = dict(zip(yolov5l_head_num, tada_conv_yolov5l_head_num))
        check_model = head
    elif back_or_head == 'headThermal':
        tada_conv_yolov5l_head_num = [str(model_num) for model_num in range(47,61)]
        yolov5l_head_num =  [str(i) for i in range(10,24)]
        head_map = dict(zip(yolov5l_head_num, tada_conv_yolov5l_head_num))
        check_model = head
        
    src_converted = {}

    for k, v in src.items():
        if k[:8] not in check_model and back_or_head == 'backbone':
            continue
        if k[:9] not in check_model and back_or_head in ['head', 'headRGB', 'headThermal']:
            continue
        if 'conv3d' in k:  # this is conv3d layer in Focus module for Tadconv version
            continue
        if ('model.0.conv.conv.weight' == k or 'model.5.conv.conv.weight'==k)  and not tadaconv:
            #takes care of concatnated CFT since we put varying sizes
            continue

        
        if back_or_head == 'backbone':
            yolov5l_model_num = k[6:7] # we know backbone of yolov5l is 0-9 so know exact location
            layer_id_tadaconv = str(backbone_map[yolov5l_model_num]) #mapped layer id
        elif back_or_head in ['head', 'headRGB', 'headThermal']:
            yolov5l_model_num = k[6:8] # we know backbone of yolov5l is 10-23 so know exact location
            layer_id_tadaconv = str(head_map[yolov5l_model_num]) #mapped layer id

        if len(k.split('.')) == 4:
            _, _, mod, w_b = k.split('.')
            if tadaconv:
                if 'conv' in mod: # from looking at name if conv is second to last its conv weights
                    new_k = f'model.{layer_id_tadaconv}.{w_b}'
                    new_v = v.unsqueeze(0).unsqueeze(0)

                elif 'bn' in mod: # from looking at name if bn is second to last its bn related (running_mean, bias)
                    new_k = f'model.{layer_id_tadaconv}.{mod}_a.{w_b}'
                    new_v = v
            else: #CFT/Concatnated CFT
                if 'conv' in mod: # from looking at name if conv is second to last its conv weights
                    new_k = f'model.{layer_id_tadaconv}.{mod}.{w_b}'
                    new_v = v

                elif 'bn' in mod: # from looking at name if bn is second to last its bn related (running_mean, bias)
                    new_k = f'model.{layer_id_tadaconv}.{mod}.{w_b}'
                    new_v = v
            
        elif len(k.split('.')) == 5:
            _, _, cv, mod, w_b = k.split('.')
            if tadaconv:
                if 'conv' in mod: # from looking at name if conv is second to last its conv weights
                    new_k = f'model.{layer_id_tadaconv}.{cv}.{w_b}'
                    new_v = v.unsqueeze(0).unsqueeze(0)
                
                elif 'bn' in mod: # from looking at name if bn is second to last its bn related (running_mean, bias)
                    new_k = f'model.{layer_id_tadaconv}.{cv}.{mod}_a.{w_b}'
                    new_v = v
            else: #CFT/Concatnated CFT
                if 'conv' in mod: # from looking at name if conv is second to last its conv weights
                    new_k = f'model.{layer_id_tadaconv}.{cv}.{mod}.{w_b}'
                    new_v = v
                
                elif 'bn' in mod: # from looking at name if bn is second to last its bn related (running_mean, bias)
                    new_k = f'model.{layer_id_tadaconv}.{cv}.{mod}.{w_b}'
                    new_v = v
        elif len(k.split('.')) == 6:
            _, _, temp1, temp2, mod, w_b = k.split('.')
            if tadaconv:
                if 'conv' in mod: # from looking at name if conv is second to last its conv weights
                    new_k = f'model.{layer_id_tadaconv}.{temp1}.{temp2}.{w_b}'
                    new_v = v.unsqueeze(0).unsqueeze(0)
                
                elif 'bn' in mod: # from looking at name if bn is second to last its bn related (running_mean, bias)
                    new_k = f'model.{layer_id_tadaconv}.{temp1}.{temp2}.{mod}_a.{w_b}'
                    new_v = v
            else: #CFT/Concatnated CFT
                if 'conv' in mod: # from looking at name if conv is second to last its conv weights
                    new_k = f'model.{layer_id_tadaconv}.{temp1}.{temp2}.{mod}.{w_b}'
                    new_v = v
                
                elif 'bn' in mod: # from looking at name if bn is second to last its bn related (running_mean, bias)
                    new_k = f'model.{layer_id_tadaconv}.{temp1}.{temp2}.{mod}.{w_b}'
                    new_v = v
        elif len(k.split('.')) == 7:
            _, _, m, num , cv, mod, w_b = k.split('.')
            if tadaconv:
                if 'conv' in mod: # from looking at name if conv is second to last its conv weights
                    new_k = f'model.{layer_id_tadaconv}.{m}.{num}.{cv}.{w_b}'
                    new_v = v.unsqueeze(0).unsqueeze(0)
                
                elif 'bn' in mod:  # from looking at name if bn is second to last its bn related (running_mean, bias)
                    new_k = f'model.{layer_id_tadaconv}.{m}.{num}.{cv}.{mod}_a.{w_b}'
                    new_v = v
            else: #CFT/Concatnated CFT
                if 'conv' in mod: # from looking at name if conv is second to last its conv weights
                    new_k = f'model.{layer_id_tadaconv}.{m}.{num}.{cv}.{mod}.{w_b}'
                    new_v = v
                
                elif 'bn' in mod:  # from looking at name if bn is second to last its bn related (running_mean, bias)
                    new_k = f'model.{layer_id_tadaconv}.{m}.{num}.{cv}.{mod}.{w_b}'
                    new_v = v

        src_converted[new_k] = new_v

    for k, v in src_converted.items():
        if k in tgt.keys():
            if not tgt[k].shape == v.shape:
                logger.info(f"Size mismatch for converting from yolov5: should be {tgt[k].shape} for {k} instead of {v.shape}")
        else:
            logger.info(f"Didn't match any keys for {k}")
    return  src_converted


    
def intersect_dicts_tadaconv(src, tgt, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    # skip = [f'model.{i}.' for i in range(10,47)]
    # skip += [f'model.{i}.' for i in [5,6,7,8]]
    save = [f'model.{i}.' for i in range(0,5)]
    save += [f'model.{9}.']
    save_set = set(save)  
    
    onlymodel9toload = [
                "model.9.cv1.bn.num_batches_tracked",
                "model.9.cv2.bn.num_batches_tracked",
                "model.9.cv3.bn.num_batches_tracked",
                "model.9.m.0.cv1.bn.num_batches_tracked",
                "model.9.m.0.cv2.bn.num_batches_tracked",
                "model.9.m.1.cv1.bn.num_batches_tracked",
                "model.9.m.1.cv2.bn.num_batches_tracked",
                "model.9.m.2.cv1.bn.num_batches_tracked",
                "model.9.m.2.cv2.bn.num_batches_tracked"
                ]
    onlymodel9toload = set(onlymodel9toload)

    src_converted = {}

    for k, v in src.items():
        if k[:8] not in save_set:
            continue
        if k[:8] == 'model.9.':
            if k not in onlymodel9toload:
                continue
        if 'rf_func' in k:
            continue
        if 'conv3d' in k:  # this is conv3d layer in Focus module
            continue
        
        if len(k.split('.')) == 4:
            _, layer_id, mod, w_b = k.split('.')
            if 'conv' in mod: # from looking at name if conv is second to last its conv weights
                new_k = f'model.{layer_id}.{w_b}'
                new_v = v.unsqueeze(0).unsqueeze(0)
            elif 'bn' in mod: # from looking at name if bn is second to last its bn related (running_mean, bias)
                new_k = f'model.{layer_id}.{mod}_a.{w_b}'
                new_v = v
            
        elif len(k.split('.')) == 5:
            _, layer_id, cv, mod, w_b = k.split('.')
            
            if 'conv' in mod: # from looking at name if conv is second to last its conv weights
                new_k = f'model.{layer_id}.{cv}.{w_b}'
                new_v = v.unsqueeze(0).unsqueeze(0)
            
            elif 'bn' in mod: # from looking at name if bn is second to last its bn related (running_mean, bias)
                new_k = f'model.{layer_id}.{cv}.{mod}_a.{w_b}'
                new_v = v
        elif len(k.split('.')) == 6:
            _, layer_id, temp1, temp2, mod, w_b = k.split('.')
            
            if 'conv' in mod: # from looking at name if conv is second to last its conv weights
                new_k = f'model.{layer_id}.{temp1}.{temp2}.{w_b}'
                new_v = v.unsqueeze(0).unsqueeze(0)
            
            elif 'bn' in mod: # from looking at name if bn is second to last its bn related (running_mean, bias)
                new_k = f'model.{layer_id}.{temp1}.{temp2}.{mod}_a.{w_b}'
                new_v = v
            
        elif len(k.split('.')) == 7:
            _, layer_id, m, num , cv, mod, w_b = k.split('.')
            
            if 'conv' in mod: # from looking at name if conv is second to last its conv weights
                new_k = f'model.{layer_id}.{m}.{num}.{cv}.{w_b}'
                new_v = v.unsqueeze(0).unsqueeze(0)
            
            elif 'bn' in mod: # from looking at name if bn is second to last its bn related (running_mean, bias)
                new_k = f'model.{layer_id}.{m}.{num}.{cv}.{mod}_a.{w_b}'
                new_v = v
        src_converted[new_k] = new_v

    for k, v in src_converted.items():
        if k in tgt.keys():
            if not tgt[k].shape == v.shape:
                logger.info(f"Size mismatch for converting from yolov5: should be {tgt[k].shape} for {k} instead of {v.shape}")
        else:
            logger.info(f"Didn't match any keys for {k}")
    return  src_converted
        


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # Finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    # Return global model sparsity
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent
    print(' %.3g global sparsity' % sparsity(model))


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False, img_size=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPS
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = ', %.1f GFLOPS' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPS
    except (ImportError, Exception):
        fs = ''

    logger.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def load_classifier(name='resnet101', n=2):
    # Loads a pretrained model reshaped to n-class output
    model = torchvision.models.__dict__[name](pretrained=True)

    # ResNet model properties
    # input_size = [3, 224, 224]
    # input_space = 'RGB'
    # input_range = [0, 1]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Reshape output to n classes
    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
