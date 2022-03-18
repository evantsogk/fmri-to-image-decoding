from utils import save_image_grid, cross_corr, ssim
import numpy as np
from scipy.stats import pearsonr
import imageio
import torch
import wandb


def train_discriminator(discriminator, criterion, optimizer, real_images, fake_images):
    """
    Trains the discriminator for one step.
    """
    optimizer.zero_grad()

    d_real = discriminator(real_images)
    real_labels = torch.ones_like(d_real)  # real labels are ones
    loss_real = criterion(d_real, real_labels)

    d_fake = discriminator(fake_images)
    fake_labels = torch.zeros_like(d_fake)  # fake labels are zeros
    loss_fake = criterion(d_fake, fake_labels)

    loss = (loss_real + loss_fake) / 2  # discriminator loss is the average of real and fake losses
    loss.backward()
    optimizer.step()

    return loss.item()


def train_generator(discriminator, criterion, pixelwise_criterion, lambda_pixel, optimizer, fake_images, real_images):
    """
    Trains the generator for one step.
    """
    optimizer.zero_grad()

    d_fake = discriminator(fake_images)
    real_labels = torch.ones_like(d_fake)
    gen_loss = criterion(d_fake, real_labels)  # uses predictions of fake images with real labels (it's a trick)
    pixel_loss = pixelwise_criterion(fake_images, real_images)  # image reconstruction error

    loss = gen_loss + lambda_pixel * pixel_loss  # add weighted pixelwise loss to the generator's loss
    loss.backward()
    optimizer.step()

    return loss.item()


def train(train_dl, test_dl, generator, discriminator, gen_optimizer, disc_optimizer, gen_criterion, disc_criterion,
          pixelwise_criterion, device, args):

    # enable cudnn autotuner to speed up training
    if device != 'cpu':
        torch.backends.cudnn.benchmark = True

    steps = 0  # count training steps
    gif_images = []  # store images to make a gif

    # start training
    generator.train()
    discriminator.train()
    wandb.watch(generator)
    wandb.watch(discriminator)
    for epoch in range(1, args.epochs + 1):
        gen_costs = disc_costs = 0.0
        for i, (fmri, real_images) in enumerate(train_dl):
            steps += 1
            real_images = real_images.to(device)
            fmri = fmri.to(device)

            # train the discriminator
            fake_images = generator(fmri)
            disc_loss = train_discriminator(discriminator=discriminator, criterion=disc_criterion,
                                             optimizer=disc_optimizer, real_images=real_images, fake_images=fake_images)
            disc_costs += disc_loss

            # train the generator
            fake_images = generator(fmri)  # need to call again
            gen_loss = train_generator(discriminator=discriminator, criterion=gen_criterion,
                                        pixelwise_criterion=pixelwise_criterion, lambda_pixel=args.lambda_pixel,
                                        optimizer=gen_optimizer, fake_images=fake_images, real_images=real_images)
            gen_costs += gen_loss

            # update wandb
            wandb.log({"Epoch": epoch, "Generator Loss": gen_loss, "Discriminator Loss": disc_loss})

            # store images to make a gif
            if steps % args.gen_interval == 0:
                generated_images = []
                with torch.no_grad():
                    for i, (fmri, image) in enumerate(test_dl):
                        fmri = fmri.to(device)
                        generated_images.extend((generator(fmri).cpu().detach() + 1) / 2)  # go from [-1, 1] to [0, 1]
                    gif_images.append(save_image_grid(generated_images))

        # print losses
        print("Epoch: {}  -  gen_loss: {}  -  disc_loss: {}".format(epoch, gen_costs / len(train_dl),
                                                                    disc_costs / len(train_dl)))

    # save final results
    test_images = []
    generated_images = []
    corrs = []
    pears = []
    ssims = []
    with torch.no_grad():
        for i, (fmri, image) in enumerate(test_dl):
            test_images.extend((image + 1) / 2)
            fmri = fmri.to(device)
            generated_images.extend((generator(fmri.to(device)).cpu().detach() + 1) / 2)

            # calculate metrics
            generated_img = generator(fmri).cpu().detach()
            res = cross_corr(image, generated_img)
            corrs.append(res)
            similarity_idx = ssim(generated_img, image)
            ssims.append(similarity_idx)
            pear = pearsonr(generated_img.squeeze().cpu().numpy().flatten(), image.squeeze().cpu().numpy().flatten())[0]
            pears.append(pear)

        test_grid = save_image_grid(test_images, args.save_path / 'test_images.png')
        gen_grid = save_image_grid(generated_images, args.save_path / 'gen_images{}.png'.format(epoch))

        wandb.log({"Test Samples": wandb.Image(test_grid, caption="Epoch: {}".format(epoch))})
        wandb.log({"Reconstructed Samples": wandb.Image(gen_grid, caption="Epoch: {}".format(epoch))})

    print('Cross-correlation: {:.4f}\nPearson corr. coef.: {:.4f}\nSSIM: {:.4f}'.format(np.mean(corrs), np.mean(pears),
                                                                                        np.mean(ssims)))

    # save the generated images as GIF file
    imageio.mimsave(args.save_path / 'gen_images.gif', gif_images)

    # save models
    torch.save(generator.state_dict(), args.save_path / 'generator.pt')
    torch.save(discriminator.state_dict(), args.save_path / 'discriminator.pt')
