import os
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

from ConSinGAN.config import get_arguments
import ConSinGAN.functions as functions
import ConSinGAN.models as models
from ConSinGAN.imresize import imresize, imresize_to_shape


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def generate_samples(netG, reals_shapes, noise_amp, scale_w=1.0, scale_h=1.0, reconstruct=False, n=50):
    if reconstruct:
        reconstruction = netG(fixed_noise, reals_shapes, noise_amp)
        if opt.train_mode == "generation" or opt.train_mode == "retarget":
            functions.save_image('{}/reconstruction.jpg'.format(dir2save), reconstruction.detach())
            functions.save_image('{}/real_image.jpg'.format(dir2save), reals[-1].detach())
        elif opt.train_mode == "harmonization":
            functions.save_image('{}/harmonization_wo_mask.jpg'.format(dir2save), reconstruction.detach())
            functions.save_image('{}/real_image.jpg'.format(dir2save), imresize_to_shape(real, reals_shapes[-1][2:], opt).detach())
        return reconstruction

    if scale_w == 1. and scale_h == 1.:
        dir2save_parent = os.path.join(dir2save, "random_samples")
    else:
        reals_shapes = [[r_shape[0], r_shape[1], int(r_shape[2]*scale_h), int(r_shape[3]*scale_w)] for r_shape in reals_shapes]
        dir2save_parent = os.path.join(dir2save, "random_samples_scale_h_{}_scale_w_{}".format(scale_h, scale_w))

    make_dir(dir2save_parent)

    for idx in range(n):
        noise = functions.sample_random_noise(opt.train_stages - 1, reals_shapes, opt)
        sample = netG(noise, reals_shapes, noise_amp)
        functions.save_image('{}/gen_sample_{}.jpg'.format(dir2save_parent, idx), sample.detach())


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--model_dir', help='input image name', required=True)
    parser.add_argument('--gpu', type=int, help='which GPU', default=0)
    parser.add_argument('--num_samples', type=int, help='which GPU', default=50)
    parser.add_argument('--naive_img', help='naive input image  (harmonization or editing)', default="")

    opt = parser.parse_args()
    _gpu = opt.gpu
    _naive_img = opt.naive_img
    __model_dir = opt.model_dir
    opt = functions.load_config(opt)
    opt.gpu = _gpu
    opt.naive_img = _naive_img
    opt.model_dir = __model_dir

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)
        opt.device = "cuda:{}".format(opt.gpu)

    dir2save = os.path.join(opt.model_dir, "Evaluation")
    make_dir(dir2save)

    print("Loading models...")
    netG = torch.load('%s/G.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    fixed_noise = torch.load('%s/fixed_noise.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    reals = torch.load('%s/reals.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    noise_amp = torch.load('%s/noise_amp.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    reals_shapes = [r.shape for r in reals]

    if opt.train_mode == "generation" or opt.train_mode == "retarget":

        print("Generating Samples...")
        with torch.no_grad():
            # # generate reconstruction
            generate_samples(netG, reals_shapes, noise_amp, reconstruct=True)

            # generate random samples of normal resolution
            rs0 = generate_samples(netG, reals_shapes, noise_amp, n=opt.num_samples)

            # generate random samples of different resolution
            generate_samples(netG, reals_shapes, noise_amp, scale_w=2, scale_h=1, n=opt.num_samples)
            generate_samples(netG, reals_shapes, noise_amp, scale_w=1, scale_h=2, n=opt.num_samples)
            generate_samples(netG, reals_shapes, noise_amp, scale_w=2, scale_h=2, n=opt.num_samples)

    elif opt.train_mode == "harmonization":
        opt.mode = "harmonization"
        real = functions.read_image_dir(opt.naive_img, opt)
        real = imresize_to_shape(real, reals_shapes[0][2:], opt)
        fixed_noise[0] = real

        out = generate_samples(netG, reals_shapes, noise_amp, reconstruct=True)

        mask_file_name = '{}_mask{}'.format(opt.naive_img[:-4], opt.naive_img[-4:])
        if os.path.exists(mask_file_name):
            opt.mode = "harmonization"
            mask = functions.read_image_dir(mask_file_name, opt)
            if mask.shape[3] != out.shape[3]:
                mask = imresize_to_shape(mask, [out.shape[2], out.shape[3]], opt)
            mask = functions.dilate_mask(mask, opt)
            out = (1 - mask) * reals[-1] + mask * out
            functions.save_image('{}/harmonization_w_mask.jpg'.format(dir2save), out.detach())
        else:
            print("Warning: mask {} not found.".format(mask_file_name))
            print("Harmonization only performed without mask.")

    elif opt.train_mode == "editing":
        opt.mode = "editing"
        noise_scaling = 0.1
        real = functions.read_image(opt)
        real = functions.adjust_scales2image(real, opt)
        reals = functions.create_reals_pyramid(real, opt)

        reals = reals[-opt.train_scales:]
        reals_shapes = [r.shape for r in reals]
        print(reals_shapes)

        fixed_noise[0] = reals[0] + noise_scaling * functions.generate_noise([opt.nc_im, reals_shapes[0][2],
                                                                              reals_shapes[0][3]],
                                                                              device=opt.device, opt=opt).detach()

        out = generate_samples(netG, reals_shapes, reconstruct=True)

        mask = functions.read_image_dir('%s/%s_mask%s' % (opt.input_dir, opt.input_name[:-4], opt.input_name[-4:]), opt)
        mask = functions.dilate_mask(mask, opt)
        out = (1 - mask) * real + mask * out
        functions.save_image('{}/edited_w_mask.jpg'.format(dir2save), out.detach())

    print("Done. Results saved at: {}".format(dir2save))

