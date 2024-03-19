"""
Main file for training DDPM on the OCTID dataset for retinal OCT image denoising. 
First the dataset gets loaded, then the DDPM is created and trained. The final model is then stored in the provided filepath.

@author: Lukas Gandler
"""

from typing import Optional
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import StepLR, LRScheduler
from torch.utils.data import DataLoader

from dataloading import load_processed_OCT
from networks.DiffusionModel import DDPM
from networks.DDPM_Net import Model


def load_checkpoint(model: DDPM, optimizer: Optimizer, scheduler: LRScheduler, filepath: str):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    losses = checkpoint['losses']
    epoch = checkpoint['epoch'] + 1     # We want to start with the next epoch and not repeat the already completed/checkpointed one
    return model, optimizer, scheduler, losses, epoch

def save_checkpoint(model: DDPM, optimizer: Optimizer, scheduler: LRScheduler, losses: list, epoch: int, checkpoint_path: str) -> None:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'losses': losses,
        'epoch': epoch,
    }, checkpoint_path)

def train(ddpm: DDPM, data_loader: DataLoader, num_epochs: int, optimizer: Optimizer, scheduler: LRScheduler,  store_path: str, checkpoint_path: Optional[str] = None):
    """
    Trains the DDPM on the given dataset and stores the trained model in the provided path.
    :param ddpm: Network to train
    :param data_loader: Dataloader for training dataset
    :param num_epochs: Number of epochs to train
    :param optimizer: Optimizer to use
    :param scheduler: Learning rate scheduler
    :param store_path: Path to store trained network
    :param checkpoint_path: Optional path to the checkpoint from which training should be continued
    :return:
    """

    criterion = nn.MSELoss()
    best_loss = float('inf')
    losses = list()
    start_epoch = 0

    # load checkpoint if one is provided
    if checkpoint_path is not None:
        print(f'Loading checkpoint {checkpoint_path}')
        ddpm, optimizer, scheduler, losses, start_epoch = load_checkpoint(ddpm, optimizer, scheduler, checkpoint_path)

    ddpm.train()
    for epoch in tqdm(range(start_epoch, num_epochs), position=1, desc=f'Training process', colour='#00ff00'):
        epoch_loss = 0.0
        for batch in tqdm(data_loader, leave=False, position=0, desc=f'Epoch {epoch+1}/{num_epochs}', colour='#005500'):
            x0 = batch[0].to(ddpm.device)  # we are only interested in the images and not the labels
            batch_size = x0.shape[0]

            # Generate noise for each image according to random timestep
            step_tensor = torch.randint(0, ddpm.num_steps, (batch_size,)).to(ddpm.device)
            noisy_images, noise = ddpm.forward(x0, step_tensor)

            # Predict noise
            noise_prediction = ddpm.backward(noisy_images, step_tensor) # step_tensor.reshape(batch_size, -1))

            # Compute loss and optimize
            loss = criterion(noise_prediction, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running loss
            epoch_loss = epoch_loss + loss.item() * batch_size / len(data_loader.dataset)

        log_string = f'Loss at epoch {epoch + 1}: {epoch_loss:.3f}'
        losses.append(epoch_loss)
        scheduler.step()

        # Store best model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            save_checkpoint(ddpm, optimizer, scheduler, losses, epoch, store_path)
            log_string += ' -- best score (stored)'

        # Checkpointing
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch+1}_loss_{epoch_loss:.4f}.pth'
            save_checkpoint(ddpm, optimizer, scheduler, losses, epoch, checkpoint_path)
            log_string += f' -- checkpoint at epoch {epoch+1}'

        print(log_string)

def main(store_model_path):
    # Define hyperparameters
    # Note: we can lower the number of training epochs due to the aggressive learning rate scheduler:
    # in the original implementation the learning rate at the end of the 500 epochs is 3.94430e^-35(!) -> could as well be zero
    # already after 100 epochs it would be around 4.76837e^-11 and after 30 epochs around 7.8125e-07
    batch_size = 1      # too few GPU memory available for higher batch-sizes
    num_epochs = 300    # lr ends up around 4.8828125e-08 at the end

    print(f'Hyperparameters:')
    print(f'\t- batch size:    {batch_size}')
    print(f'\t- num. epochs:   {num_epochs}')

    # Load dataset
    print(f'Loading data')
    # data_loader = load_OCT(batch_size, num_workers=1)
    data_loader = load_processed_OCT(batch_size, num_workers=1)

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\t' + (f'{torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'cpu'))

    # Create DDPM -> values and model taken from hu et al. (https://arxiv.org/pdf/2201.11760.pdf)
    n_steps = 100
    min_beta = 0.0001
    max_beta = 0.006
    ddpm = DDPM(Model(), n_steps, min_beta, max_beta, device)

    # Training
    optimizer = Adam(ddpm.parameters(), lr=1e-4, betas=(0.5, 0.999))
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)      # paper used 3000 images, we have 500 -> must need 6-times the updates before adjusting lr
    train(ddpm, data_loader, num_epochs, optimizer, scheduler, store_model_path)

    print(f'Training done - Best Model stored at {store_model_path}')


if __name__ == '__main__':
    store_model_path = 'OCT_model_BEST.pt'
    main(store_model_path)
