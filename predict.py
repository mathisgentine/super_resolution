import image_slicer
import argparse
import os
import shutil
import sys

import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *
from torchvision.models import vgg16_bn

from tqdm import tqdm
from PIL import Image

#For LAMB optimizer

import collections
import math

import torch
from tensorboardX import SummaryWriter
from torch.optim import Optimizer

import warnings
warnings.filterwarnings("ignore")

#Define class FeatureLoss
class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()

def log_lamb_rs(optimizer: Optimizer, event_writer: SummaryWriter, token_count: int):
    """Log a histogram of trust ratio scalars in across layers."""
    results = collections.defaultdict(list)
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            for i in ('weight_norm', 'adam_norm', 'trust_ratio'):
                if i in state:
                    results[i].append(state[i])

    for k, v in results.items():
        event_writer.add_histogram(f'lamb/{k}', torch.tensor(v), token_count)

class Lamb(Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0, adam=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        self.adam = adam
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(group['weight_decay'], p.data)

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(-step_size * trust_ratio, adam_step)

        return loss

#ArgumentParser
parser = argparse.ArgumentParser()
parser.add_argument("File_name", help="Path of the image you want to apply the model on",
                    type=str)
parser.add_argument("Split_size", help="Number of tiles the image will be split in",
                    type=int)
parser.add_argument("Zoom_type", help="0 for \"4x20x\", 1 for \"20x100x\"",
                    type=int)
args = parser.parse_args()

file_name = args.File_name
num_tiles = args.Split_size
zoom_type = args.Zoom_type

zoom_dict = {0: "4x20x", 1: "20x100x"}

#Load learner
LEARNER_FOLDER = os.getcwd() + f'/{zoom_dict[zoom_type]}-{num_tiles}'
learn = load_learner(LEARNER_FOLDER)

#Split image
DEST_PATH = LEARNER_FOLDER + '/image_split'

if not os.path.exists(DEST_PATH):
    os.makedirs(DEST_PATH)

tiles = image_slicer.slice(file_name, num_tiles, save=False)
image_slicer.save_tiles(tiles, directory=DEST_PATH, format='tiff', prefix='split')

#Add test folder
learn.data.add_test(ImageImageList.from_folder(DEST_PATH), label=None, tfms=None, tfm_y=False)

#Sort data_test & files #Very important
files = sorted(os.listdir(DEST_PATH))
learn.data.test_ds.x.items = sorted(learn.data.test_ds.x.items)

##Predict

#Iterate over every image, save in a few folder
NEW_PATH = LEARNER_FOLDER + '/predicted_image_split'

if not os.path.exists(NEW_PATH):
    os.makedirs(NEW_PATH)

for index, f in tqdm(enumerate(learn.data.test_ds.x.items)):
    new_el = learn.predict(learn.data.test_ds[index][0])[0] #FastAI image type
    ending_name = files[index].split('split')[1] #Corresponding file name in DEST_PATH
    new_el.save(NEW_PATH + f'/pred{ending_name}')

#Reconstitute image
tiles = image_slicer.open_images_in(NEW_PATH)
big_tiles = image_slicer.join(tiles)

#Delete image_split in the file
shutil.rmtree(DEST_PATH)
shutil.rmtree(NEW_PATH)

##Show Images & Save
#Test image
img = Image.open(file_name)
img.show()

#Prediction
big_tiles.show()

#Save
file_name_last = file_name.split('/')[-1]
big_tiles.save(LEARNER_FOLDER + '/' + file_name_last)
