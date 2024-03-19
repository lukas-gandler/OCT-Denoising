"""
A sample implementation OCT denoising using the trained model on the RETOUCH dataset.

@author: Lukas Gandler
"""

import argparse
import SimpleITK as sitk
import numpy as np
import cv2 as cv
from typing import Tuple

import torch
import torchvision.transforms as transforms
from networks.DDPM_Net import Model
from networks.DiffusionModel import DDPM


def normalize_image(image: np.ndarray) -> np.ndarray:
    return ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)


def load_retouch_image(file: str) -> np.ndarray:
    image = sitk.ReadImage(file)
    image_array = sitk.GetArrayFromImage(image)

    image_idx = image_array.shape[0] // 2  # take the middle of the volume
    extracted_image = image_array[image_idx]
    extracted_image = normalize_image(extracted_image)  # have to normalize image to the expected uint8 representation
    return extracted_image


def denoise_image(model: DDPM, image: np.ndarray, steps=100) -> Tuple[np.ndarray, np.ndarray]:
    trans = transforms.Compose([
        transforms.ToTensor(),  # here we need a specific image data type (we use uint8)
        transforms.Grayscale(),
        transforms.Pad(100, padding_mode='reflect'), # pad edges with reflect to replicate noisy pattern when images are too small
        transforms.CenterCrop((512, 512)),
        transforms.Normalize(mean=[0.21008788933651848], std=[0.1335889231399768]),
    ])

    input_image = trans(image)[None, :]
    denoised_image = model.denoise(input_image.float().to(model.device), int(steps)).detach().cpu().numpy().squeeze()
    denoised_image = normalize_image(denoised_image)
    denoised_image = cv.putText(denoised_image, f'output (t={steps})', (0, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

    input_image = normalize_image(input_image.detach().cpu().numpy().squeeze())
    input_image = cv.putText(input_image, f'input', (0, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
    return denoised_image, input_image


def main(file_path: str, denoising_steps: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model_file = f'../OCT_model.pt'
    n_steps, min_beta, max_beta = 100, 0.0001, 0.006

    ddpm = DDPM(Model(), n_steps, min_beta, max_beta, device)
    ddpm.load_state_dict(torch.load(model_file))
    ddpm.to(device)

    retouch_image = load_retouch_image(file_path)
    denoised_image, input_image = denoise_image(ddpm, retouch_image, denoising_steps)

    result = np.vstack((input_image, denoised_image))
    cv.imshow('result', result)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes as input the oct.mhd file and the number of denosing steps')
    parser.add_argument('-f', '--file', type=str, required=True, help='The path to the oct.mhd file')
    parser.add_argument('-s', '--steps', type=int, required=True, help='The number of denoising steps (must be int <= 100)')

    args = vars(parser.parse_args())
    main(args['file'], args['steps'])
