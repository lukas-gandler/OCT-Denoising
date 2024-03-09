"""
Implementation of a DDPM in which an arbitrary network can be inserted.

@author: Lukas Gandler
"""

import torch
import torch.nn as nn


class DDPM(nn.Module):
    def __init__(self, network: nn.Module, num_steps: int, min_beta: float, max_beta: float, device: torch.device):
        """
        DDPM class that takes care of forward as well as backward process independent of provided network. It also
        implements methods for generating and denoising data.
        :param network: The network structure of the model (typically some U-Net)
        :param num_steps: The length of the Markov process
        :param min_beta: The minimum beta (NOTE: uses linear scheduling strategy)
        :param max_beta: The maximum beta (NOTE: uses linear scheduling strategy)
        :param device: Device to operate on
        """
        super(DDPM, self).__init__()

        self.num_steps = num_steps
        self.device = device
        self.network = network.to(device)

        # TODO: Introduce some kind of strategy pattern to accommodate for different beta schedules if needed
        self.betas = torch.linspace(min_beta, max_beta, num_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward(self, x0: torch.Tensor, t: torch.Tensor): # -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes t-th image in the forward Markov process through closed form solution based on x0 and alphas
        :param x0: Ground truth image
        :param t: Time-step
        :return: (t-th element in the markov chain, added noise)
        """
        batch_size = x0.shape[0]
        a_bar = self.alpha_bars[t]
        noise = torch.randn_like(x0).to(self.device)

        # TODO: according to the paper-implementation this should use alpha_bars instead of a_bar
        xt = a_bar.sqrt().reshape(batch_size, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(batch_size, 1, 1, 1) * noise
        return xt, noise

    def backward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Runs the input through the network to predict the added noise
        :param x: Corrupted image
        :param t: Timestep
        :return: Noise estimation
        """
        return self.network(x, t)

    @torch.no_grad()
    def generate_new_images(self, num_samples: int = 5, image_shape = (1, 28, 28)) -> torch.Tensor:
        """
        Generates new images
        :param num_samples: The number of images to be generated
        :param image_shape: The shape of the images
        :param t_start: Optional: Custom starting point
        :return: Generated images
        """

        # Generate random noise
        channels, height, width = image_shape
        noise = torch.randn(num_samples, channels, height, width).to(self.device)

        # Generate images from noise
        generated_images = self.denoise(noise)
        return generated_images

    @torch.no_grad()
    def denoise(self, images: torch.Tensor, t_start: int = None) -> torch.Tensor:
        """
        Iteratively denoises the given images
        :param images: Corrupted imges to denoise
        :param t_start: Optional: Determines the starting point
        :return: Denoised images
        """

        num_samples, channels, height, width = images.shape
        start_point = self.num_steps if t_start is None else t_start
        for idx, t in enumerate(reversed(range(start_point))):
            # Estimate noise to be removed
            step_tensor = torch.tensor([t] * num_samples).to(self.device).long()
            noise_prediction = self.backward(images, step_tensor)

            # Denoise images
            alpha_t = self.alphas[t]
            alpha_t_bar = self.alpha_bars[t]
            images = (1 / alpha_t.sqrt()) * (images - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * noise_prediction)

            if t > 0:
                # Add more noise -> Langevin Dynamics
                z = torch.randn(num_samples, channels, height, width).to(self.device)
                sigma_t = self.betas[t].sqrt()
                images = images + sigma_t * z

        return images

    @torch.no_grad()
    def hu_denoising(self, images, t):
        """
        Re-implements the denosing process in Hu et al. as they did not use the iterative approach
        :param images: Input images
        :param t: Determines the starting point
        :return: Denoised images
        """

        t = t - 1

        num_samples, _, _, _ = images.shape
        step_tensor = torch.tensor([t] * num_samples).to(self.device).long()
        eps_t = self.backward(images, step_tensor)

        alpha_t = self.alphas[t]
        alpha_t_bar = self.alpha_bars[t]

        x0_pred = 1.0 / alpha_t_bar.sqrt() * images - (1 - alpha_t_bar).sqrt() / alpha_t_bar.sqrt() * eps_t
        return x0_pred
