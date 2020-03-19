import os
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

from ConSinGAN.config import get_arguments
import ConSinGAN.functions as functions
import ConSinGAN.models as models


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def init_model(opt, shape):
    #generator initialization:
    netG = models.GrowingGeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    for _ in range(opt.stop_scale):
        netG.init_next_stage()

    return netG


noise_scaling = 0.1
def generate_samples(netG, reals_shapes, m_pad, m_pad_block, noise_amp, scale_w=1.0, scale_h=1.0, reconstruct=False, n=50):
    if reconstruct:
        reconstruction = netG(fixed_noise, reals_shapes, m_pad, m_pad_block, noise_amp)
        functions.save_image('{}/reconstruction.jpg'.format(dir2save), reconstruction.detach())
        functions.save_image('{}/real_image.jpg'.format(dir2save), reals[-1].detach())
        return

    if scale_w == 1. and scale_h == 1.:
        dir2save_parent = os.path.join(dir2save, "random_samples")
    else:
        reals_shapes = [[r_shape[0], r_shape[1], int(r_shape[2]*scale_h), int(r_shape[3]*scale_w)] for r_shape in reals_shapes]
        dir2save_parent = os.path.join(dir2save, "random_samples_scale_h_{}_scale_w_{}".format(scale_h, scale_w))

    make_dir(dir2save_parent)

    for idx in range(n):
        noise = []
        for d in range(opt.train_stages):
            if d == 0:
                noise.append(functions.generate_noise([opt.nc_im, reals_shapes[d][2], reals_shapes[d][3]],
                                                      device=opt.device).detach())
            else:
                noise.append(functions.generate_noise([opt.nfc,
                                                       reals_shapes[d][2] + opt.num_layer * 2,
                                                       reals_shapes[d][3] + opt.num_layer * 2],
                                                      device=opt.device).detach())
        sample = netG(noise, reals_shapes, m_pad, m_pad_block, noise_amp)
        functions.save_image('{}/gen_sample_{}.jpg'.format(dir2save_parent, idx), sample.detach())



if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--model_dir', help='input image name', required=True)
    parser.add_argument('--gpu', type=int, help='which GPU', default=0)

    num_samples = 10

    opt = parser.parse_args()
    _gpu = opt.gpu

    if not os.path.exists(opt.model_dir):
        print("Model not found: {}".format(opt.model_dir))
        exit()

    with open(os.path.join(opt.model_dir, 'parameters.txt'), 'r') as f:
        params = f.readlines()
        for param in params:
            param = param.split("-")
            param = [p.strip() for p in param]
            param_name = param[0]
            param_value = param[1]
            try:
                param_value = int(param_value)
            except:
                ValueError
            setattr(opt, param_name, param_value)
    opt.gpu = _gpu

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

    if opt.train_mode == "generation":
        m_pad = nn.ZeroPad2d(1)
        m_pad_block = nn.ZeroPad2d(2)

        print("Generating Samples...")
        with torch.no_grad():
            # # generate reconstruction
            generate_samples(netG, reals_shapes, m_pad, m_pad_block, noise_amp, reconstruct=True)

            # generate random samples of normal resolution
            rs0 = generate_samples(netG, reals_shapes, m_pad, m_pad_block, noise_amp,)

            # generate random samples of different resolution
            sh = [2, 2, 1]
            sw = [1, 2, 2]
            generate_samples(netG, reals_shapes, m_pad, m_pad_block, noise_amp, scale_w=sw[0], scale_h=sh[0])
            generate_samples(netG, reals_shapes, m_pad, m_pad_block, noise_amp, scale_w=sw[1], scale_h=sh[1])
            generate_samples(netG, reals_shapes, m_pad, m_pad_block, noise_amp, scale_w=sw[2], scale_h=sh[2])
    print("Done. Results saved at: {}".format(dir2save))
    exit()

    # set other values
    m_pad = nn.ZeroPad2d(1)
    m_pad_block = nn.ZeroPad2d(3)

    # with torch.no_grad():
    #     # # generate reconstruction
    #     # generate_samples(G_curr, opt.stop_scale, start_scale=0, reconstruct=True)
    #
    #     # generate random samples from scale 0
    #     # rs0 = generate_samples(G_curr, opt.stop_scale, start_scale=0, return_samples=num_samples)
    #
    #     # generate random samples from scale 1
    #     # rs1 = generate_samples(G_curr, opt.stop_scale, start_scale=1, return_samples=num_samples)
    #
    #     # generate random samples of different sizes
    #     # sv = [1.7, 1.8, 1.9, 1, 1, 1]
    #     # sh = [1, 1, 1, 1.7, 1.8, 1.9]
    #     sv = [2.0, 1.0, 2.0, 1, 1, 1]
    #     sh = [1, 2.0, 2.0, 1, 1, 1]
    #     rs0_12 = generate_samples(G_curr, opt.stop_scale, scale_v=sv[0], scale_h=sh[0], return_samples=num_samples)
    #     rs0_21 = generate_samples(G_curr, opt.stop_scale, scale_v=sv[1], scale_h=sh[1], return_samples=num_samples)
    #     rs0_22 = generate_samples(G_curr, opt.stop_scale, scale_v=sv[2], scale_h=sh[2], return_samples=num_samples)
    #
    #     # rs0_105 = generate_samples(G_curr, opt.stop_scale, scale_v=sv[3], scale_h=sh[3], return_samples=num_samples)
    #     # rs0_051 = generate_samples(G_curr, opt.stop_scale, scale_v=sv[4], scale_h=sh[4], return_samples=num_samples)
    #     # rs0_0505 = generate_samples(G_curr, opt.stop_scale, scale_v=sv[5], scale_h=sh[5], return_samples=num_samples)
    #
    #     # superresolution
    #     # generate_samples(G_curr, opt.stop_scale, start_scale=0, scale_v=2, scale_h=2, reconstruct=True)

    # generate quick overview image of samples
    # grid_rows = 8
    # grid_columns = num_samples
    # fig, grid = create_image_grid(grid_rows, grid_columns)
    # fill_image_grid(grid, rs0, 0, name="random samples 0")
    # fill_image_grid(grid, rs1, 1, name="random samples 1")
    #
    # fill_image_grid(grid, rs0_12, 2, name="random samples 0-12")
    # fill_image_grid(grid, rs0_21, 3, name="random samples 0-21")
    # fill_image_grid(grid, rs0_22, 4, name="random samples 0-22")
    #
    # fill_image_grid(grid, rs0_105, 5, name="random samples 0-105")
    # fill_image_grid(grid, rs0_051, 6, name="random samples 0-051")
    # fill_image_grid(grid, rs0_0505, 7, name="random samples 0-0505")
    #
    # plt.savefig("{}/random_samples.jpg".format(dir2save))
    # plt.close()


    # generate animation

    # use edited image
    if opt.edit:
        # def read_image(path):
        #     x = img.imread(path)
        #     x = functions.np2torch(x, opt)
        #     x = x[:, 0:3, :, :]
        #     return x
        reals = []
        real = functions.read_image(opt)
        # functions.adjust_scales2image(real, opt)
        real = imresize(real, opt.scale1, opt)
        reals = functions.creat_reals_pyramid(real, reals, opt)
        # print(len(Zs))
        # print(len(reals))
        # print(reals[0].shape)
        # exit()
        # Zs[0] = reals[0]
        # start_scale = opt.start_scale
        reals = reals[-opt.train_scales:]
        reals_shapes = [r.shape for r in reals]
        print(reals_shapes)
        # exit()
        # Zs_rec = [functions.generate_noise([filters,reals_shapes[d][2],reals_shapes[d][3]], device=opt.device).detach() for d in range(len(Zs))]
        # Zs_rec = [z for z in Zs_rec[start_scale:]]
        # print(reals_shapes)
        # exit()
        # _inp_img = read_image("/data/hinz/SinGAN/Input/Images/pisa1_edit.jpg")
        Zs_rec[0] = reals[0] + noise_scaling * functions.generate_noise([3, _reals_shapes[0][2], _reals_shapes[0][3]], device=opt.device, opt=opt).detach()

        # Zs_rec[0] = functions.imresize_to_shape(_inp_img, reals_shapes[0][2:], opt)

        # print([z.shape for z in Zs_rec])
        # print(reals_shapes)
        # exit()
        # print([r.shape for r in reals])
        # print([z.shape for z in Zs_rec])
        # exit()

        if opt.img_norm == 1:
            Zs_rec[0] = functions.np2torch_img_norm(opt, reals[0].shape[2:])
        print([z.shape for z in Zs_rec])
        # exit()
        out = generate_samples(G_curr, opt.stop_scale, start_scale=0, reconstruct=True)
        # print(out.shape, opt.stop_scale)
        # exit()

        opt.mode = "editing"
        # opt.mode = "harmonization"
        print("Opt Mode is: {}".format(opt.mode))
        opt.ref_dir = opt.input_dir
        opt.ref_name = opt.input_name
        mask = functions.read_image_dir('%s/%s_mask%s' % (opt.input_dir, opt.input_name[:-4], opt.input_name[-4:]), opt)
        if mask.shape[3] != real.shape[3]:
            from SinGAN.imresize import imresize_to_shape
            mask = imresize_to_shape(mask, [real.shape[2], real.shape[3]], opt)
            mask = mask[:, :, :real.shape[2], :real.shape[3]]
            # ref = imresize_to_shape(mask, [real.shape[2], real.shape[3]], opt)
            # ref = ref[:, :, :real.shape[2], :real.shape[3]]
        mask = functions.dilate_mask(mask, opt)
        # print(mask.shape)
        # print(out.shape)
        # print(reals_shapes)
        # exit()
        out = (1 - mask) * real + mask * out
        plt.imsave('{}/reconstruction_w_mask.jpg'.format(dir2save), functions.convert_image_np(out.detach(), opt),
                   vmin=0, vmax=1)
