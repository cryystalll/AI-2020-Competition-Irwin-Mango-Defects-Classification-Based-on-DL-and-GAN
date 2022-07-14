from __future__ import print_function

import argparse
import math
import os
import pprint
import random
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils

from generator import Generator
from discriminator import Discriminator


# Parse arguments.
def my_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="want to try something but don't want to cover old files"
    )
    parser.add_argument(
        "--manual_seed",
        type=int,
        help="manual seed"
    )
    parser.add_argument(
        "--dataroot",
        help="path to dataset"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="the height / width of the input image to network, default=64"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size, default=64"
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="number of data loading workers, default=2",
        default=2
    )
    parser.add_argument(
        "--ngf",
        type=int,
        default=64,
        help="number of filters in the generator, default=64"
    )
    parser.add_argument(
        "--extra_layers",
        type=int,
        default=0,
        help="extra layers, default=0"
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=100,
        help="size of the latent z vector, default=100"
    )
    parser.add_argument(
        "--G_pth",
        type=int,
        help="index of path to Generator net (to continue training)"
    )
    parser.add_argument(
        "--checkpoint",
        default=".",
        help="folder to output model checkpoints"
    )
    parser.add_argument(
        "--ndf",
        type=int,
        default=64,
        help="number of filters in the discriminator, default=64"
    )
    parser.add_argument(
        "--D_pth",
        type=int,
        help="index of path to Discriminator net (to continue training)"
    )
    parser.add_argument(
        "--dlr",
        type=float,
        default=0.0002,
        help="discriminator learning rate, default=0.0002"
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.5,
        help="beta1 for adam. default=0.5"
    )
    parser.add_argument(
        "--glr",
        type=float,
        default=0.0002,
        help="generator learning rate, default=0.0002"
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=25,
        help="number of epochs to train for, default=25"
    )
    parser.add_argument(
        "--bl_D_x",
        type=float,
        default=0.5,
        help="balance value for D_x, default=0.5"
    )
    parser.add_argument(
        "--bl_D_G_z1",
        type=float,
        default=0.5,
        help="balance value for D_G_z1, default=0.5"
    )
    parser.add_argument(
        "--bl_D_G_z2",
        type=float,
        default=0.5,
        help="balance value for D_G_z2, default=0.5"
    )
    parser.add_argument(
        "--real_image",
        default=".",
        help="folder to output Real images"
    )
    parser.add_argument(
        "--fake_image",
        default=".",
        help="folder to output Fake images"
    )

    return parser.parse_args()


# Read and load data.
def make_dataset_and_dataloader(dataroot, image_size, batch_size, workers):
    dataset = datasets.ImageFolder(
        root=dataroot,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(workers)
    )
    return (dataset, dataloader)


def weights_initial(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def main():
    # Get arguments.
    argument = my_parser()

    # for test
    test_path = "test/" if argument.test is True else ""

    # Set manual seed.
    if argument.manual_seed is None:
        argument.manual_seed = random.randint(1, 10000)
    print(f"Random Seed: {argument.manual_seed}")
    random.seed(argument.manual_seed)
    torch.manual_seed(argument.manual_seed)

    # Set CPU/GPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ngpu = torch.cuda.device_count()

    # TODO: Check what this really does!
    cudnn.benchmark = True

    # Read and load data.
    (dataset, dataloader) = make_dataset_and_dataloader(
        argument.dataroot,
        argument.image_size,
        argument.batch_size,
        argument.workers
    )
    data_length = len(dataloader)

    # Set net of generator.
    netG = Generator(
        ngpu,
        argument.image_size,
        argument.ngf,
        argument.extra_layers,
        argument.nz
    ).to(device)
    weights_initial(netG)
    if argument.G_pth is not None:
        netG.load_state_dict(
            torch.load(
                f"{argument.checkpoint}/generator_net_{argument.G_pth}.pth")
        )
    pprint.pprint(netG)

    # Set net of discriminator.
    netD = Discriminator(
        ngpu,
        argument.image_size,
        argument.ndf,
        argument.extra_layers
    ).to(device)
    weights_initial(netD)
    if argument.D_pth is not None:
        netD.load_state_dict(
            torch.load(
                f"{argument.checkpoint}/discriminator_net_{argument.D_pth}.pth"
            )
        )
    pprint.pprint(netD)

    # TODO: Check what this really does!
    criterion = nn.BCELoss()

    # Create noise.
    fixed_noise = torch.randn(
        argument.batch_size,
        argument.nz,
        1,
        1,
        device=device
    )

    # Set True/False label.
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(
        netD.parameters(),
        lr=argument.dlr,
        betas=(argument.beta1, 0.999)
    )
    optimizerG = optim.Adam(
        netG.parameters(),
        lr=argument.glr,
        betas=(argument.beta1, 0.999)
    )

    image_dict = {i: j for (j, i) in dataset.class_to_idx.items()}
    for epoch in range(1, argument.niter+1):
        for i, data in enumerate(dataloader, 0):
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full(
                (batch_size,),
                real_label,
                device=device
            )

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, argument.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake

            if (D_x <= argument.bl_D_x or D_G_z1 >= argument.bl_D_G_z1):
                optimizerD.step()
            else:
                print("Skip D")

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            if D_G_z2 <= argument.bl_D_G_z2:
                optimizerG.step()
            else:
                print("Skip G")

            # Show information.
            print(f"[{epoch}/{argument.niter}][{i}/{len(dataloader)}]",
                  f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}",
                  f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
                  )

            if (i + 1) % (data_length//125) == 0:
                # Save the image.
                utils.save_image(
                    real_cpu,
                    f"{argument.real_image}/{test_path}real.jpg",
                    nrow=int(math.sqrt(batch_size - 1)) + 1,
                    normalize=True
                )
                fake = netG(fixed_noise)
                utils.save_image(
                    fake.detach(),
                    f"{argument.fake_image}/{test_path}" +
                    f"fake_{epoch}_{(i+1)//(data_length//125)}.jpg",
                    nrow=int(math.sqrt(batch_size - 1)) + 1,
                    normalize=True
                )

        # Save model.
        torch.save(
            netG.state_dict(),
            f"{argument.checkpoint}/{test_path}generator_net_{epoch}.pth")
        torch.save(
            netD.state_dict(),
            f"{argument.checkpoint}/{test_path}discriminator_net_{epoch}.pth")


if __name__ == '__main__':
    main()
