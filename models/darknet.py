import torch
import torch.nn as nn
import numpy as np
import struct
from collections import OrderedDict
import torch.nn.functional as F

def parse_cfg(cfg_path):
    """Parse YOLO cfg file and return module definitions"""
    with open(cfg_path, 'r') as f:
        lines = f.read().split('\n')
        lines = [x for x in lines if x and not x.startswith('#')]
        lines = [x.rstrip().lstrip() for x in lines]
        
    blocks = []
    block = {}
    for line in lines:
        if line.startswith('['):
            if block:
                blocks.append(block)
            block = {'type': line[1:-1].rstrip()}
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    
    return blocks

class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_size=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.img_size = img_size
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}

    def forward(self, x):
        return x

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class Darknet(nn.Module):
    def __init__(self, cfg_path):
        super(Darknet, self).__init__()
        self.module_defs = parse_cfg(cfg_path)
        self.hyperparams = self.module_defs.pop(0)  # Remove hyperparams
        self.module_list = self.create_modules()
        self.yolo_layers = [layer[-1] for layer in self.module_list if isinstance(layer[-1], YOLOLayer)]

    def create_modules(self):
        """Create module list of layer blocks from module configuration in module_defs"""
        output_filters = [3]  # Initial filters (RGB)
        module_list = nn.ModuleList()
        
        for i, mdef in enumerate(self.module_defs):
            modules = nn.Sequential()
            
            if mdef['type'] == 'convolutional':
                filters = int(mdef['filters'])
                kernel_size = int(mdef['size'])
                pad = (kernel_size - 1) // 2 if int(mdef.get('pad', 0)) else 0
                modules.add_module(
                    'conv_%d' % i,
                    nn.Conv2d(in_channels=output_filters[-1],
                             out_channels=filters,
                             kernel_size=kernel_size,
                             stride=int(mdef.get('stride', 1)),
                             padding=pad,
                             bias='batch_normalize' not in mdef))
                
                if 'batch_normalize' in mdef:
                    modules.add_module('bn_%d' % i, nn.BatchNorm2d(filters))
                if mdef.get('activation') == 'leaky':
                    modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1, inplace=True))
                output_filters.append(filters)
            
            elif mdef['type'] == 'maxpool':
                kernel_size = int(mdef['size'])
                stride = int(mdef['stride'])
                modules.add_module('pool_%d' % i, nn.MaxPool2d(kernel_size=kernel_size, stride=stride))
                output_filters.append(output_filters[-1])
            
            elif mdef['type'] == 'route':
                layers = [int(x) for x in mdef['layers'].split(',')]
                filters = sum([output_filters[i if i > 0 else len(output_filters) + i] for i in layers])
                modules.add_module('route_%d' % i, EmptyLayer())
                output_filters.append(filters)
            
            elif mdef['type'] == 'shortcut':
                filters = output_filters[int(mdef['from'])]
                modules.add_module('shortcut_%d' % i, EmptyLayer())
                output_filters.append(filters)
            
            elif mdef['type'] == 'yolo':
                anchor_idxs = [int(x) for x in mdef['mask'].split(',')]
                anchors = [float(x) for x in mdef['anchors'].split(',')]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in anchor_idxs]
                num_classes = int(mdef['classes'])
                img_size = int(self.hyperparams['width'])
                modules.add_module('yolo_%d' % i, YOLOLayer(anchors, num_classes, img_size))
                output_filters.append(output_filters[-1])
            
            module_list.append(modules)
            
        return module_list

    def load_darknet_weights(self, weights_path):
        """Load YOLO weights from file"""
        # Open the weights file
        with open(weights_path, 'rb') as f:
            # First five values are header information
            header = np.fromfile(f, dtype=np.int32, count=5)
            
            # The rest of the values are weights
            weights = np.fromfile(f, dtype=np.float32)
        
        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if 'batch_normalize' in module_def:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

        return self

    def forward(self, x):
        layer_outputs = []
        output = []

        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            
            if mtype in ['convolutional', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layers = [int(x) for x in mdef['layers'].split(',')]
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                else:
                    try:
                        x = torch.cat([layer_outputs[i if i > 0 else len(layer_outputs) + i] for i in layers], 1)
                    except:  # apply stride 2 for darknet reorg layer
                        layer_outputs[layers[1]] = F.interpolate(layer_outputs[layers[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([layer_outputs[i if i > 0 else len(layer_outputs) + i] for i in layers], 1)
            elif mtype == 'shortcut':
                from_layer = int(mdef['from'])
                x = layer_outputs[-1] + layer_outputs[from_layer if from_layer > 0 else len(layer_outputs) + from_layer]
            elif mtype == 'yolo':
                x = module(x)
                output.append(x)
            layer_outputs.append(x)

        return output if self.training else torch.cat(output, 1)
