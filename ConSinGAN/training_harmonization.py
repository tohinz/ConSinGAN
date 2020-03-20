
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

from ConSinGAN.imresize import imresize, imresize_to_shape
import ConSinGAN.functions as functions
import ConSinGAN.models as models

def train(opt):
    print("Training model with the following parameters:")
    print("\t number of stages: {}".format(opt.train_stages))
    print("\t number of concurrently trained stages: {}".format(opt.train_depth))
    print("\t learning rate scaling: {}".format(opt.lr_scale))
    print("\t non-linearity: {}".format(opt.activation))

    real = functions.read_image(opt)
    real = functions.adjust_scales2image(real, opt)
    reals = functions.create_reals_pyramid(real, opt)
    print("Training on image pyramid: {}".format([r.shape for r in reals]))
    print("")

    if opt.naive_img != "":
        _input_name = opt.input_name
        opt.input_name = opt.naive_img
        naive_img = functions.read_image(opt)
        naive_img = imresize_to_shape(naive_img, reals[0].shape[2:], opt)
        naive_img = functions.convert_image_np(naive_img)*255.0
        opt.input_name = _input_name
    else:
        naive_img = None

    if opt.fine_tune:
        img_to_augment = naive_img
    else:
        img_to_augment = functions.convert_image_np(reals[0])*255.0

    generator = init_G(opt)
    fixed_noise = []
    noise_amp = []

    for scale_num in range(opt.stop_scale+1):
        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_,scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
                print(OSError)
                pass
        functions.save_image('{}/real_scale.jpg'.format(opt.outf), reals[scale_num])

        d_curr = init_D(opt)
        if scale_num > 0:
            d_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_, scale_num - 1)))
            generator.init_next_stage()

        writer = SummaryWriter(log_dir=opt.outf)
        fixed_noise, noise_amp, generator, d_curr = train_single_scale(d_curr, generator, reals, img_to_augment,
                                                                       naive_img, fixed_noise, noise_amp,
                                                                       opt, scale_num, writer)

        torch.save(fixed_noise, '%s/fixed_noise.pth' % (opt.out_))
        torch.save(generator, '%s/G.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(noise_amp, '%s/noise_amp.pth' % (opt.out_))
        del d_curr
    writer.close()
    return



def train_single_scale(netD, netG, reals, img_to_augment, naive_img, fixed_noise, noise_amp, opt, depth, writer):
    reals_shapes = [real.shape for real in reals]
    real = reals[depth]
    aug = functions.Augment()

    alpha = opt.alpha

    ############################
    # define z_opt for training on reconstruction
    ###########################
    if depth == 0:
        z_opt = reals[0]
    else:
        z_opt = functions.generate_noise([opt.nfc, reals_shapes[depth][2], reals_shapes[depth][3]],
                                         device=opt.device)
    fixed_noise.append(z_opt.detach())

    ############################
    # define optimizers, learning rate schedulers, and learning rates for lower stages
    ###########################
    # setup optimizers for D
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

    # setup optimizers for G
    # remove gradients from stages that are not trained
    for block in netG.body[:-opt.train_depth]:
        for param in block.parameters():
            param.requires_grad = False

    # set different learning rate for lower stages
    parameter_list = [
        {"params": block.parameters(), "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-opt.train_depth:]) - 1 - idx))}
        for idx, block in enumerate(netG.body[-opt.train_depth:])]

    # add parameters of head and tail to training
    if depth - opt.train_depth < 0:
        parameter_list += [{"params": netG.head.parameters(), "lr": opt.lr_g * (opt.lr_scale ** depth)}]
    parameter_list += [{"params": netG.tail.parameters(), "lr": opt.lr_g}]
    optimizerG = optim.Adam(parameter_list, lr=opt.lr_g, betas=(opt.beta1, 0.999))

    # define learning rate schedules
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[0.8*opt.niter], gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[0.8*opt.niter], gamma=opt.gamma)

    ############################
    # calculate noise_amp
    ###########################
    if depth == 0:
        noise_amp.append(1)
    else:
        noise_amp.append(0)
        z_reconstruction = netG(fixed_noise, reals_shapes, noise_amp)

        criterion = nn.MSELoss()
        rec_loss = criterion(z_reconstruction, real)

        RMSE = torch.sqrt(rec_loss).detach()
        _noise_amp = opt.noise_amp_init * RMSE
        noise_amp[-1] = _noise_amp

    # start training
    _iter = tqdm(range(opt.niter))
    for iter in _iter:
        _iter.set_description('scale %d:[%d/%d]' % (depth, iter+1, opt.niter))

        ############################
        # (0) sample augmented training image
        ###########################
        noise = []
        for d in range(depth + 1):
            if d == 0:
                if opt.fine_tune:
                    noise.append(reals_add[0])
                else:
                    data = {"image": img_to_augment}
                    augmented = aug.transform(**data)
                    image = augmented["image"]
                    noise.append(functions.np2torch(image, opt))
            else:
                noise.append(functions.generate_noise([opt.nfc, reals_shapes[d][2], reals_shapes[d][3]],
                                                      device=opt.device).detach())

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            netD.zero_grad()

            output = netD(real)
            errD_real = -output.mean()

            # train with fake
            if j == opt.Dsteps -1:
                fake = netG(noise, reals_shapes, noise_amp)
            else:
                with torch.no_grad():
                    fake = netG(noise, reals_shapes, noise_amp)

            output = netD(fake.detach())
            errD_fake = output.mean()

            gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
            errD_total = errD_real + errD_fake + gradient_penalty
            errD_total.backward()
            optimizerD.step()

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################
        output = netD(fake)
        errG = -output.mean()

        if alpha != 0:
            loss = nn.MSELoss()
            rec = netG(fixed_noise, reals_shapes, noise_amp)
            rec_loss = alpha * loss(rec, real)
        else:
            rec_loss = 0

        netG.zero_grad()
        errG_total = errG + rec_loss
        errG_total.backward()

        for _ in range(opt.Gsteps):
            optimizerG.step()

        ############################
        # (3) Log Results
        ###########################
        if iter % 250 == 0 or iter + 1 == opt.niter:
            writer.add_scalar('Loss/train/D/real/{}'.format(j), -errD_real.item(), iter + 1)
            writer.add_scalar('Loss/train/D/fake/{}'.format(j), errD_fake.item(), iter + 1)
            writer.add_scalar('Loss/train/D/gradient_penalty/{}'.format(j), gradient_penalty.item(), iter + 1)
            writer.add_scalar('Loss/train/G/gen', errG.item(), iter + 1)
            writer.add_scalar('Loss/train/G/reconstruction', rec_loss.item(), iter + 1)
        if iter % 500 == 0 or iter + 1 == opt.niter:
            functions.save_image('{}/fake_sample_{}.jpg'.format(opt.outf, iter + 1), fake.detach())
            functions.save_image('{}/reconstruction_{}.jpg'.format(opt.outf, iter + 1), rec.detach())
            generate_samples(netG, img_to_augment, naive_img, aug, opt, depth, noise_amp, writer, reals, iter + 1)

        schedulerD.step()
        schedulerG.step()
        # break

    functions.save_networks(netG, netD, z_opt, opt)
    return fixed_noise, noise_amp, netG, netD


def generate_samples(netG, img_to_augment, naive_img, aug, opt, depth, noise_amp, writer, reals, iter, n=16):
    opt.out_ = functions.generate_dir2save(opt)
    dir2save = '{}/harmonized_samples_stage_{}'.format(opt.out_, depth)
    reals_shapes = [r.shape for r in reals]
    images = []
    try:
        os.makedirs(dir2save)
    except OSError:
        pass

    if naive_img is not None:
        n = n-1
    with torch.no_grad():
        for idx in range(n):
            noise = []
            for d in range(depth + 1):
                if d == 0:
                    if opt.fine_tune:
                        noise.append(reals_add[0])
                    else:
                        data = {"image": img_to_augment}
                        augmented = aug.transform(**data)
                        augmented_image = functions.np2torch(augmented["image"], opt)
                        noise.append(augmented_image)
                else:
                    noise.append(functions.generate_noise([opt.nfc, reals_shapes[d][2], reals_shapes[d][3]],
                                                          device=opt.device).detach())
            sample = netG(noise, reals_shapes, noise_amp)
            functions.save_image('{}/{}_naive_sample.jpg'.format(dir2save, idx), augmented_image)
            functions.save_image('{}/{}_harmonized_sample.jpg'.format(dir2save, idx), sample.detach())
            augmented_image = imresize_to_shape(augmented_image, sample.shape[2:], opt)
            images.append(augmented_image)
            images.append(sample.detach())

        if opt.fine_tune:
            _input_dir, _input_name = opt.input_dir, opt.input_name
            _mode = opt.mode
            opt.input_dir, opt.input_name = "Input/Harmonization", opt.harmonization_img

            harmonized_img_original_size = functions.read_image(opt)
            harmonized_img_original = imresize_to_shape(harmonized_img_original_size, Zs[-1].shape[2:], opt)
            harmonized_img_original_resized = imresize_to_shape(harmonized_img_original_size, Zs[0].shape[2:], opt)
            Zs_harmonization = [z for z in Zs]
            Zs_harmonization[0] = harmonized_img_original_resized
            harmonized_img = netG(Zs_harmonization, scale, reals_shapes, m_pad, m_pad_block, NoiseAmp)

            opt.mode = "harmonization"
            opt.ref_dir = opt.input_dir
            opt.ref_name = opt.input_name
            mask = functions.read_image_dir('%s/%s_mask%s' % (opt.input_dir, opt.input_name[:-4], opt.input_name[-4:]), opt)
            if mask.shape[3] != original_img.shape[3]:
                mask = imresize_to_shape(mask, [original_img.shape[2], original_img.shape[3]], opt)
                mask = mask[:, :, :original_img.shape[2], :original_img.shape[3]]
            mask = functions.dilate_mask(mask, opt)
            harmonized_img_w_mask = (1 - mask) * original_img + mask * harmonized_img

            harmonize_imgs = torch.cat([original_img, reconstruction_img, harmonized_img_original, harmonized_img, harmonized_img_w_mask], 0)
            grid_2 = make_grid(harmonize_imgs, nrow=5, normalize=True)
            writer.add_image('rec_harmon_harmon_w_mask_{}'.format(scale), grid_2, 0)

            if epoch is not None:
                plt.imsave('{}/harmonized_img_w_mask_{}.png'.format(opt.outf, epoch), functions.convert_image_np(harmonized_img_w_mask.detach(), opt),
                           vmin=0, vmax=1)

            opt.input_dir, opt.input_name = _input_dir, _input_name
            opt.mode = _mode
        else:
            if naive_img is not None:
                noise = []
                for d in range(depth + 1):
                    if d == 0:
                        noise.append(functions.np2torch(naive_img, opt))
                    else:
                        noise.append(functions.generate_noise([opt.nfc, reals_shapes[d][2], reals_shapes[d][3]],
                                                              device=opt.device).detach())
                sample = netG(noise, reals_shapes, noise_amp)
                sample = add_mask(sample, opt)
                _naive_img = imresize_to_shape(functions.np2torch(naive_img, opt), sample.shape[2:], opt)
                images.insert(0, sample.detach())
                images.insert(0, _naive_img)

            images = torch.cat(images, 0)
            grid = make_grid(images, nrow=4, normalize=True)
            writer.add_image('harmonized_images_{}'.format(depth), grid, iter)


def add_mask(img, opt):
    return img

def init_G(opt):
    # generator initialization:
    netG = models.GrowingGenerator(opt).to(opt.device)
    netG.apply(models.weights_init)
    # print(netG)

    return netG


def init_D(opt):
    #discriminator initialization:
    netD = models.Discriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    # print(netD)

    return netD
