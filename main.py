from dataset import FMRIDataset, load_brains_data, load_69_fmri, load_neuron
from models import Generator, Discriminator
from trainer import train
from utils import calculate_elapsed_time
import glob
import argparse
from pathlib import Path
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchsummary import summary
import wandb


def parse_args():
    """
    Configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=Path, required=True)  # path to save logs and checkpoints
    parser.add_argument("--data_path", type=Path, required=True)  # path to data

    parser.add_argument("--lambda_pixel", type=float, default=150)  # lambda value for the feature matching loss
    parser.add_argument("--batch_size", type=int, default=1)  # batch size
    parser.add_argument("--gen_lr", type=float, default=0.0002)  # generator learning rate
    parser.add_argument("--disc_lr", type=float, default=0.0002)  # discriminator learning rate

    parser.add_argument("--epochs", type=int, default=25)  # number of epochs to train
    parser.add_argument("--gen_interval", type=int, default=30)  # (in steps) how often to generate samples
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.save_path.mkdir(parents=True, exist_ok=True)  # create save directory if it doesn't exist

    # wandb configuration
    wandb.init(project='biodata_final', entity='dsml-team')
    # wandb.init(mode="disabled")
    config = wandb.config
    config.lambda_pixel = args.lambda_pixel

    # save code to wandb
    code = wandb.Artifact('code', type='code')
    for path in glob.glob('**/*.py', recursive=True):
        if path.find("main.py") == -1:  # adding main throws exception
            code.add_file(path)
    wandb.run.use_artifact(code)

    # get default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    if "69" in str(args.data_path):
        train_fmri, test_fmri, train_images, test_images = load_69_fmri(args.data_path)
    elif "brains" in str(args.data_path):
        train_fmri, test_fmri, train_images, test_images = load_brains_data(args.data_path)
    else:
        train_fmri, test_fmri, train_images, test_images = load_neuron(args.data_path)

    # create datasets
    img_size = 112
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size), transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_dataset = FMRIDataset(train_fmri, train_images, transform=transform)
    test_dataset = FMRIDataset(test_fmri, test_images, transform=transform)

    # create loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # print shapes
    print("fMRI shape:", train_fmri.shape)
    print("Images shape:", train_images.shape)
    train_features, train_labels = next(iter(train_loader))
    print(f"fMRI batch shape: {train_features.size()}")
    print(f"Images batch shape: {train_labels.size()}")
    print("Device:", device)

    # models
    generator = Generator(train_fmri.shape[1]).to(device)
    discriminator = Discriminator(1).to(device)

    # print model summaries
    summary(generator.cuda(), (train_fmri.shape[1],))
    summary(discriminator.cuda(), (1, img_size, img_size))

    # optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=args.gen_lr, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=args.disc_lr, betas=(0.5, 0.999))

    # criterions
    gen_criterion = nn.BCEWithLogitsLoss()
    disc_criterion = nn.BCEWithLogitsLoss()
    pixelwise_criterion = nn.MSELoss()

    # train
    start = time.time()
    train(train_dl=train_loader, test_dl=test_loader, generator=generator, discriminator=discriminator,
          gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer, gen_criterion=gen_criterion,
          disc_criterion=disc_criterion, pixelwise_criterion=pixelwise_criterion, device=device, args=args)
    end = time.time()
    print("Training time:", calculate_elapsed_time(start, end))
