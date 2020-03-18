import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ConSinGAN.imresize import imresize, imresize_to_shape
import ConSinGAN.functions as functions
import ConSinGAN.models as models


def train(opt, Gs, Zs, reals, NoiseAmp):
    print("Training model with the following parameters:")
    print("\t number of stages: {}".format(opt.train_stages))
    print("\t number of concurrently trained stages: {}".format(opt.train_depth))
    print("\t learning rate scaling: {}".format(opt.lr_scale))
    # print("\t")
    print("")

    real_ = functions.read_image(opt)
    scale_num = 0
    real = imresize(real_, opt.scale1, opt)
    reals = functions.create_reals_pyramid(real, reals, opt)
    reals = reals[opt.start_scale:]

    m_pad = nn.ZeroPad2d(1)
    m_pad_block = nn.ZeroPad2d(2)

    print("Image pyramid:")
    print([_real.shape for _real in reals])
    G_curr = init_G(opt)
    opt.nzx = reals[0].shape[2]
    opt.nzy = reals[0].shape[3]

    while scale_num<opt.stop_scale+1-opt.start_scale:
        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_,scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
                print("error")
                print(OSError)
                pass
        plt.imsave('%s/real_scale.png' % (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

        D_curr = init_D(opt)
        if scale_num > 0:
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_,scale_num-1)))
            G_curr.init_next_stage()

        writer = SummaryWriter(log_dir=opt.outf)
        Zs, NoiseAmp, G_curr, D_curr = train_single_scale(D_curr, G_curr, reals, Zs, NoiseAmp, opt, scale_num, writer)
        generate_samples(G_curr, opt, scale_num, m_pad, m_pad_block, NoiseAmp, writer, reals)

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))
        scale_num+=1
        del D_curr
    writer.close()
    return


def train_single_scale(netD, netG, reals, Zs, NoiseAmp, opt, depth, writer):
    reals_shapes = [real.shape for real in reals]
    real = reals[depth]
    opt.nzx = real.shape[2]
    opt.nzy = real.shape[3]

    m_pad = nn.ZeroPad2d(1)
    m_pad_block = nn.ZeroPad2d(2)


    alpha = opt.alpha

    # define z_opt for training on reconstruction
    if depth == 0:
        z_opt = reals[0]
    else:
        z_opt = functions.generate_noise([opt.nfc,
                                          reals_shapes[depth][2]+opt.num_layer*2,
                                          reals_shapes[depth][3]+opt.num_layer*2],
                                          device=opt.device)

    Zs.append(z_opt.detach())

    # setup optimizers for D
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

    # setup optimizers for G
    # remove gradients from stages that are not trained
    for block in netG.body[:-opt.train_depth]:
        for param in block.parameters():
            param.requires_grad = False

    # set different learning rate for lower stages
    parameter_list = [{"params": block.parameters(), "lr": opt.lr_g * (opt.lr_scale**(len(netG.body[-opt.train_depth:])-1-idx))}
               for idx, block in enumerate(netG.body[-opt.train_depth:])]

    # add parameters of head and tail to training
    if depth - opt.train_depth < 0:
        parameter_list += [{"params": netG.head.parameters(), "lr": opt.lr_g * (opt.lr_scale**depth)}]
    parameter_list += [{"params": netG.tail.parameters(), "lr": opt.lr_g}]
    optimizerG = optim.Adam(parameter_list, lr=opt.lr_g, betas=(opt.beta1, 0.999))

    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[0.8*opt.niter], gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[0.8*opt.niter], gamma=opt.gamma)

    _iter = tqdm(range(opt.niter))
    
    for iter in _iter:
        _iter.set_description('stage [%d/%d]:[%d/%d]' % (depth, opt.stop_scale, iter+1, opt.niter))

        # sample noise for unconditional generation
        noise = []
        for d in range(depth+1):
            if d == 0:
                noise.append(functions.generate_noise([opt.nc_im, reals_shapes[d][2], reals_shapes[d][3]],
                                                          device=opt.device).detach())
            else:
                noise.append(functions.generate_noise([opt.nfc,
                                                       reals_shapes[d][2]+opt.num_layer*2,
                                                       reals_shapes[d][3]+opt.num_layer*2],
                                                       device=opt.device).detach())
        ############################
        # (0) Calculate NoiseAmp
        ###########################
        if iter == 0 and depth == 0:
            noise_amp = 1
            NoiseAmp.append(noise_amp)
        elif iter == 0:
            NoiseAmp.append(0)
            z_reconstruction = netG(Zs, reals_shapes, m_pad, m_pad_block, NoiseAmp)
            z_reconstruction = imresize_to_shape(z_reconstruction, real.shape[2:], opt)

            criterion = nn.MSELoss()
            rec_loss = criterion(z_reconstruction, real)

            RMSE = torch.sqrt(rec_loss).detach()
            noise_amp = opt.noise_amp_init * RMSE
            NoiseAmp[-1] = noise_amp

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            # train with real
            netD.zero_grad()
            output = netD(real)
            errD_real = -output.mean()

            # train with fake
            if j == opt.Dsteps - 1:
                fake = netG(noise, reals_shapes, m_pad, m_pad_block, NoiseAmp)
            else:
                with torch.no_grad():
                    fake = netG(noise, reals_shapes, m_pad, m_pad_block, NoiseAmp)

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
            rec = netG(Zs, reals_shapes, m_pad, m_pad_block, NoiseAmp)
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
        if iter % 250 == 0 or iter == (opt.niter - 1):
            writer.add_scalar('Loss/train/D/real/{}'.format(j), -errD_real.item(), iter)
            writer.add_scalar('Loss/train/D/fake/{}'.format(j), errD_fake.item(), iter)
            writer.add_scalar('Loss/train/D/gradient_penalty/{}'.format(j), gradient_penalty.item(), iter)
            writer.add_scalar('Loss/train/G/gen', errG.item(), iter)
            writer.add_scalar('Loss/train/G/reconstruction', rec_loss.item(), iter)
        if iter % 500 == 0 or iter == (opt.niter - 1):
            functions.save_image('{}/fake_sample_{}.png'.format(opt.outf, iter), fake.detach())
            functions.save_image('{}/reconstruction_{}.png'.format(opt.outf, iter), rec.detach())
            generate_samples(netG, opt, depth, m_pad, m_pad_block, NoiseAmp, writer, reals)

        schedulerD.step()
        schedulerG.step()
        # break

    functions.save_networks(netG, netD, z_opt, opt)
    return Zs, NoiseAmp, netG, netD


def generate_samples(netG, opt, scale, m_pad, m_pad_block, NoiseAmp, writer, reals, n=5):
    opt.out_ = functions.generate_dir2save(opt)
    dir2save = '{}/gen_samples_stage_{}'.format(opt.out_, scale)
    reals_shapes = [r.shape for r in reals]
    all_images = []
    try:
        os.makedirs(dir2save)
    except OSError:
        pass
    with torch.no_grad():
        for idx in range(n):
            noise = []
            for d in range(scale + 1):
                if d == 0:
                    noise.append(functions.generate_noise([opt.nc_im, reals_shapes[d][2], reals_shapes[d][3]],
                                                          device=opt.device).detach())
                else:
                    noise.append(functions.generate_noise([opt.nfc,
                                                           reals_shapes[d][2] + opt.num_layer * 2,
                                                           reals_shapes[d][3] + opt.num_layer * 2],
                                                           device=opt.device).detach())
            sample = netG(noise, reals_shapes, m_pad, m_pad_block, NoiseAmp)
            all_images.append(sample)
            functions.save_image('{}/gen_sample_{}.png'.format(dir2save, idx), sample.detach())

        all_images = torch.cat(all_images, 0)
        all_images[0] = reals[scale].squeeze()
        grid = make_grid(all_images, nrow=min(10, n), normalize=True)
        writer.add_image('gen_images_{}'.format(scale), grid, 0)


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
