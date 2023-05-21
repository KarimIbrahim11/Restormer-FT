import torch
import os
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from runpy import run_path
from skimage.util import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
import numpy as np

task = 'Motion_Deblurring'


def get_weights_and_parameters(task, parameters):
    if task == 'Motion_Deblurring':
        weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
    elif task == 'Single_Image_Defocus_Deblurring':
        weights = os.path.join('Defocus_Deblurring', 'pretrained_models', 'single_image_defocus_deblurring.pth')
    elif task == 'Deraining':
        weights = os.path.join('Deraining', 'pretrained_models', 'deraining.pth')
    elif task == 'Real_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'real_denoising.pth')
        parameters['LayerNorm_type'] = 'BiasFree'
    return weights, parameters


# Get model weights and parameters
parameters = {'inp_channels': 3, 'out_channels': 3, 'dim': 48, 'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4,
              'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66, 'bias': False, 'LayerNorm_type': 'WithBias',
              'dual_pixel_task': False}
weights, parameters = get_weights_and_parameters(task, parameters)

load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
model = load_arch['Restormer'](**parameters)
model.cuda()

checkpoint = torch.load(weights)
model.load_state_dict(checkpoint['params'])
model.eval()

input_dir = 'demo/sample_images/' + task + '/degraded'
out_dir = 'demo/sample_images/' + task + '/restored'
os.makedirs(out_dir, exist_ok=True)
extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
files = natsorted(glob(os.path.join(input_dir, '*')))

img_multiple_of = 8

print(f"\n ==> Running {task} with weights {weights}\n ")
with torch.no_grad():
    for filepath in tqdm(files):
        # print(file_)
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
        input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).cuda()

        # Pad the input if not_multiple_of 8
        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                (w + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - h if h % img_multiple_of != 0 else 0
        padw = W - w if w % img_multiple_of != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        restored = model(input_)
        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:, :, :h, :w]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        filename = os.path.split(filepath)[-1]
        cv2.imwrite(os.path.join(out_dir, filename), cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))





# Visualization

import matplotlib.pyplot as plt

inp_filenames = natsorted(glob(os.path.join(input_dir, '*')))
out_filenames = natsorted(glob(os.path.join(out_dir, '*')))

## Will display only first 5 images
num_display_images = 5
if len(inp_filenames) > num_display_images:
    inp_filenames = inp_filenames[:num_display_images]
    out_filenames = out_filenames[:num_display_images]

print(f"Results: {task}")
for inp_file, out_file in zip(inp_filenames, out_filenames):
    degraded = cv2.cvtColor(cv2.imread(inp_file), cv2.COLOR_BGR2RGB)
    restored = cv2.cvtColor(cv2.imread(out_file), cv2.COLOR_BGR2RGB)
    ## Display Images
    fig, axes = plt.subplots(nrows=1, ncols=2)
    dpi = fig.get_dpi()
    fig.set_size_inches(900 / dpi, 448 / dpi)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axes[0].axis('off')
    axes[0].imshow(degraded)
    axes[1].axis('off')
    axes[1].imshow(restored)
    plt.show()
