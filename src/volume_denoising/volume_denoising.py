import sys
sys.path.append('../')

import argparse
import numpy as np
import SimpleITK as sitk
import cv2 as cv
import traceback

import os
from tqdm import tqdm

import torch
import torchvision.transforms as transforms

from networks.DDPM_Net import Model
from networks.DiffusionModel import DDPM

def normalize_image(image):
    return ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)


def read_volume(path):
    image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(image)
    return image_array


def save_volume(image_array, path):
    image = sitk.GetImageFromArray(image_array)
    sitk.WriteImage(image, path, useCompression=False)  # save as .mhd and .raw


def pre_process_region(region):
    height, width = region.shape
    padding_up_down = (512 - height) // 2
    padding_left_right = (512 - width) // 2

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Pad([padding_left_right, padding_up_down, padding_left_right, padding_up_down], padding_mode='reflect'), # pad on all sides evenly if necessary
        transforms.Normalize(mean=[0.21008788933651848], std=[0.1335889231399768]),
    ])

    processed_region = trans(region)[None,:]    # add an extra dimension for the batch
    return processed_region


def split_into_regions(slice):
    height, width = slice.shape

    regions = list()
    placement = list()

    region1 = slice[0:min(512, height), 0:min(512, width)]
    regions.append(pre_process_region(region1))
    placement.append((0, min(512, height), 0, min(512, width)))

    # Note: I could just take 513-width/height for the other regions but I want to maximize the actual image content
    # for the denoising process and keep the padding to a minimum

    if width > 512:
        region2 = slice[0:min(512, height), width-512:width]
        regions.append(pre_process_region(region2))
        placement.append((0, min(512, height), width-512, width))

    if height > 512:
        region3 = slice[height-512:height, 0:min(512, width)]
        regions.append(pre_process_region(region3))
        placement.append((height-512, height, 0, min(512, width)))

    if width > 512 and height > 512:
        region4 = slice[height-512:height, width-512:width]
        regions.append(pre_process_region(region4))
        placement.append((height-512, height, width-512, width))

    return regions, placement

def histogram_matching(regions):
    if len(regions) == 1:
        return regions

    reference_image = normalize_image(regions[0])
    reference_hist, _ = np.histogram(reference_image, bins=256, range=[0,256])
    reference_cdf = reference_hist.cumsum()
    normalized_reference_hist = reference_cdf / reference_cdf.max()


    for idx, target_regions in enumerate(regions[1:]):
        normalized_target_regions = normalize_image(target_regions)

        target_hist, _ = np.histogram(normalized_target_regions, bins=256, range=[0,256])
        target_cdf = target_hist.cumsum()
        normalized_target_hist = target_cdf / target_cdf.max()

        mapping = np.interp(normalized_target_hist, normalized_reference_hist, range(256))
        regions[1 + idx] = mapping[normalized_target_regions]

    regions[0] = reference_image
    return regions


def denoise_bscan(bscan, steps, model):
    regions, placement = split_into_regions(bscan)

    region_tensor = torch.cat(regions, dim=0)
    denoised_regions = model.denoise(region_tensor.float().to(model.device), int(steps)).detach().cpu().numpy()
    # denoised_regions = histogram_matching(denoised_regions)       # another option to normalize the images but no noticeable improvements (also only works with taking max)

    denoised_layers = np.ones((len(regions), bscan.shape[0], bscan.shape[1])) * -np.inf

    for idx, coords in enumerate(placement):
        x_start, x_end, y_start, y_end = coords

        height_padding = (512 - (x_end - x_start)) // 2
        width_padding  = (512 - (y_end - y_start)) // 2

        region = denoised_regions[idx].squeeze()
        denoised_layers[idx, x_start:x_end, y_start:y_end] = region[height_padding:(x_end-x_start) + height_padding, width_padding:(y_end-y_start) + width_padding]

    merged_denoised_bscan = np.max(denoised_layers, axis=0)         # take max to avoid slight borders as good as possible
    merged_denoised_bscan = normalize_image(merged_denoised_bscan)

    return merged_denoised_bscan

def denoise_volume(path, steps, model):
    volume_array = read_volume(path)
    volume_array = normalize_image(volume_array)
    denoised_volume = np.zeros_like(volume_array)

    for idx, bscan in enumerate(tqdm(volume_array)):
        denoised_bscan = denoise_bscan(bscan, steps, model)
        denoised_volume[idx] = denoised_bscan

    return denoised_volume

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model_file = f'../../OCT_model.pt'
    n_steps, min_beta, max_beta = 100, 0.0001, 0.006

    ddpm = DDPM(Model(), n_steps, min_beta, max_beta, device)
    ddpm.load_state_dict(torch.load(model_file))
    ddpm.to(device)
    ddpm = torch.compile(ddpm)
    return ddpm

def main(denoising_jobs):
    model = load_model()

    with open(denoising_jobs) as fp:
        lines = fp.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            try:
                file_path, steps = line.split(' ')
                denoised_volume = denoise_volume(file_path, steps, model)

                directory, _ = os.path.split(file_path)
                denoised_file_path = os.path.join(directory, 'oct_denoised.mhd')
                save_volume(denoised_volume, denoised_file_path)

                print(f'Denoised version saved as {denoised_file_path}')
            except Exception as error:
                print(f'An error has occurred for line: {line}:')
                print(error)
                traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes a file with denoising jobs and saves denoised images into folders')
    parser.add_argument('-f', '--file', type=str, required=True, help='The path to the file with the denoising jobs (e.g. denoising_jobs.txt)')

    args = vars(parser.parse_args())
    main(args['file'])
