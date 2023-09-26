import os
from os import path
import time
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from model.eval_network import STCN
from dataset.davis_test_dataset import DAVISTestDataset
from dataset.range_transform import inv_im_trans
from util.tensor_util import unpad
from inference_core import InferenceCore
from torchvision import transforms

from progressbar import progressbar
##################################
from torch.autograd import Variable
##################################
import random
import math

################################
from attackers.region_attacker import RegionAttacker,DummyAttacker,  ground_truth_generator

# attacker = RegionAttacker().cuda()
attacker = DummyAttacker().cuda()
optimizer_attack = torch.optim.Adam(attacker.parameters(), 0.1, weight_decay=0.01)

####################################
"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/stcn.pth')
parser.add_argument('--davis_path', default='./data/Davis/DAVIS-2017/')
parser.add_argument('--output',default='demo')
parser.add_argument('--split', help='val/testdev', default='val')
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--mem_every', default=5, type=int)
parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')
parser.add_argument('--percent_part',type=float,default=1)
args = parser.parse_args()

davis_path = args.davis_path
out_path = args.output

# Simple setup
os.makedirs(out_path, exist_ok=True)

palette = Image.open(path.expanduser(davis_path + '/Annotations/480p/blackswan/00000.png')).convert('P').getpalette()


# Setup Dataset
if args.split == 'val':
    test_dataset = DAVISTestDataset(davis_path, imset='2017/val.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
elif args.split == 'testdev':
    test_dataset = DAVISTestDataset(davis_path, imset='2017/test-dev.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
else:
    raise NotImplementedError

# Load our checkpoint
top_k = args.top
prop_model = STCN().cuda().eval()

#############################################
def pad_divide_by(in_img, d, in_size=None):
    if in_size is None:
        h, w = in_img.shape[-2:]
    else:
        h, w = in_size

    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    out = F.pad(in_img, pad_array)
    return out, pad_array
##############################################
step = 3

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
upper_limit = (1. - mean) / std
lower_limit = (0. - mean) / std
upper_limit = upper_limit.reshape(1,3,1,1).repeat(3,1,1,1).cuda()
lower_limit = lower_limit.reshape(1,3,1,1).repeat(3,1,1,1).cuda()

def attack_RA(data,perturb):

    iters = 20
    epsilon = 8.0/255/std
    epsilon_step = 2.0/255/std
    i=0
    rgb = data['rgb'][:,i:i+step,:,:,:].cuda()
    perturb = torch.randn(rgb.shape).uniform_(-8.0/255,8.0/255).cuda()
    epsilon = epsilon.reshape(1,3,1,1).repeat(3,1,1,1).cuda()
    epsilon_step = epsilon_step.reshape(1,3,1,1).repeat(3,1,1,1).cuda()
    rgb_adv = rgb
    attacker_pred_at_iter = []
    for iter in range(iters):
        if iter==0: rgb_adv += perturb
        grad_list1 = []
        grad_list2 = []
        rgb_few = Variable(rgb_adv, requires_grad=True)
        msk = data['gt'][0]
        msk_few = msk[:,i:i+step,:,:,:].cuda()
        msk2 = data['gt2'][0]
        msk2_few = msk2[i:i+step,:,:].cuda()
        info = data['info']
        k = len(info['labels'][0])
        processor = InferenceCore(prop_model, rgb_few, k, top_k=top_k, 
                        mem_every=args.mem_every, include_last=args.include_last)
        processor.interact(msk_few[:,0], 0, rgb_few.shape[1], attack=True)
        gt2, pad_array = pad_divide_by(msk2_few, 16, in_size=None)
        
        input0 = processor.prob.permute(1,0,2,3,4).squeeze(2)  
        loss = F.cross_entropy(input0,gt2.long(), reduction='none')
        gt_for_attack = ground_truth_generator(loss.detach())
        loss = loss.mean()
        prop_model.zero_grad()
        loss.backward()
        
        rgb_grad =rgb_few.grad
        grad_list1.append(rgb_grad)
        # ################################################
        attack_pred = attacker(rgb_grad.detach()[:,0,...])
        attack_pred = F.interpolate(attack_pred, size=gt_for_attack.shape[-2:], mode='bilinear', align_corners=False)
        loss_attack = F.binary_cross_entropy_with_logits(attack_pred, gt_for_attack[0].unsqueeze(0).unsqueeze(0))
        optimizer_attack.zero_grad()
        loss_attack.backward()

        optimizer_attack.step()
        attack_pred = unpad(attack_pred.detach(), pad_array)
        attacker_pred_at_iter.append(attack_pred.detach()) 
        # # ##############################################
        perturbs_list = grad_list1 + grad_list2
        a = perturbs_list[0]
        for i in range(1,len(perturbs_list)):
            a = torch.cat((a, perturbs_list[i]),1)
        del loss
        del processor
        del perturbs_list
        del rgb_grad 

        first_frame_pred = torch.sigmoid( attacker_pred_at_iter[-1])
        first_frame_pred_ = first_frame_pred.repeat(1,3,1,1)

        delta = torch.tensor([item.cpu().detach().numpy() for item in a]).cuda()
        delta_sign = torch.sign(delta)

        delta_sign[:,0,...] *=  (first_frame_pred_ + 1)
        rgb_adv += epsilon_step*delta_sign.cuda()
        rgb_adv = torch.where(rgb_adv > rgb+epsilon, rgb+epsilon, rgb_adv)
        rgb_adv = torch.where(rgb_adv < rgb-epsilon, rgb-epsilon, rgb_adv)
        rgb_adv = torch.max(torch.min(rgb_adv,upper_limit),lower_limit)
    return  rgb_adv

prop_saved = torch.load(args.model)
for k in list(prop_saved.keys()):
    if k == 'value_encoder.conv1.weight':
        if prop_saved[k].shape[1] == 4:
            pads = torch.zeros((64,1,7,7), device=prop_saved[k].device)
            prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
prop_model.load_state_dict(prop_saved)

total_process_time = 0
total_frames = 0


percent_part = args.percent_part
# Start eval
for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):
    with torch.cuda.amp.autocast(enabled=args.amp):
        
        torch.autograd.set_grad_enabled(True)
        rgb_adv_temp = attack_RA(data,perturb=None)
        torch.autograd.set_grad_enabled(False)
        msk = data['gt'][0].cuda()
        height_range = range(msk.shape[3])
        height = msk.shape[3]
        width_range = range(msk.shape[4])
        width = msk.shape[4]

        rgb = data['rgb'].cuda()

        rgb_adv = torch.cat([rgb_adv_temp[:,0,...].unsqueeze(1), rgb[:,1:,...]], dim=1) 
        
        info = data['info']
        name = info['name'][0]
        k = len(info['labels'][0])
        size = info['size_480p']

        torch.cuda.synchronize()
        process_begin = time.time()

        processor = InferenceCore(prop_model, rgb_adv, k, top_k=top_k, 
                        mem_every=args.mem_every, include_last=args.include_last)
        processor.interact(msk[:,0], 0, rgb_adv.shape[1], attack=False)

        # Do unpad -> upsample to original size 
        out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
        for ti in range(processor.t):
            prob = unpad(processor.prob[:,ti], processor.pad)
            prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
            out_masks[ti] = torch.argmax(prob, dim=0)
        
        out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

        torch.cuda.synchronize()
        total_process_time += time.time() - process_begin
        total_frames += out_masks.shape[0]

        # Save the results
        this_out_path = path.join(out_path, name)
        os.makedirs(this_out_path, exist_ok=True)
        for f in range(out_masks.shape[0]):
            img_E = Image.fromarray(out_masks[f])
            img_E.putpalette(palette)
            img_E.save(os.path.join(this_out_path, '{:05d}.png'.format(f)))

        # del rgb
        del rgb_adv
        del msk
        del processor

print('Total processing time: ', total_process_time)
print('Total processed frames: ', total_frames)
print('FPS: ', total_frames / total_process_time)