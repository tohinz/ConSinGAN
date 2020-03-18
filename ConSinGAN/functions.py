import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn as nn
from torch._six import inf
import scipy.io as sio
import math
from skimage import io as img
from skimage import color, morphology, filters
from skimage.transform import rescale
#from skimage import morphology
#from skimage import filters
from SinGAN.imresize import imresize, imresize_in, imresize_to_shape
import os
import random
from sklearn.cluster import KMeans
import datetime
import dateutil.tz
from scipy.stats import truncnorm
import copy

from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize
from albumentations.pytorch import ToTensor
from albumentations import (
        HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
        Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
        IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
        IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, MultiplicativeNoise, ToSepia,
        ChannelDropout, ChannelShuffle, Cutout, InvertImg
    )

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def norm(x):
    out = (x -0.5) *2
    return out.clamp(-1, 1)

#def denorm2image(I1,I2):
#    out = (I1-I1.mean())/(I1.max()-I1.min())
#    out = out*(I2.max()-I2.min())+I2.mean()
#    return out#.clamp(I2.min(), I2.max())

#def norm2image(I1,I2):
#    out = (I1-I2.mean())*2
#    return out#.clamp(I2.min(), I2.max())

def convert_image_np(inp, opt=None):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = inp.numpy().transpose((0,1))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])

    inp = np.clip(inp,0,1)
    return inp

def save_image(real_cpu,receptive_feild,ncs,epoch_num,file_name):
    fig,ax = plt.subplots(1)
    if ncs==1:
        ax.imshow(real_cpu.view(real_cpu.size(2),real_cpu.size(3)),cmap='gray')
    else:
        #ax.imshow(convert_image_np(real_cpu[0,:,:,:].cpu()))
        ax.imshow(convert_image_np(real_cpu.cpu()))
    rect = patches.Rectangle((0,0),receptive_feild,receptive_feild,linewidth=5,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.savefig(file_name)
    plt.close(fig)

def convert_image_np_2d(inp):
    inp = denorm(inp)
    inp = inp.numpy()
    # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
    # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])
    # inp = std*
    return inp


# def truncated_normal(noise, mean=0, std=1, truncate=1):
#     # def truncated_normal_(tensor, mean=0, std=1):
#     # size = tensor.shape
#     # tmp = tensor.new_empty(size + (4,)).normal_()
#     # valid = (tmp < 2) & (tmp > -2)
#     # ind = valid.max(-1, keepdim=True)[1]
#     # tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
#     # tensor.data.mul_(std).add_(mean)
#
#     size = noise.shape
#     tmp = torch.cuda.FloatTensor(size + (4,)).normal_()
#     valid = (tmp < truncate) & (tmp > -truncate)
#     print(valid)
#     ind = valid.max(-1, keepdim=True)[1]
#     noise.data.copy_(tmp.gather(-1, ind).squeeze(-1))
#     noise.data.mul_(std).add_(mean)
#     print(noise.shape)
#     print(torch.min(noise), torch.max(noise), torch.mean(noise), torch.std(noise))
#     exit()
#     return noise


def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    values = torch.from_numpy(values).type(torch.cuda.FloatTensor)
    return values


def generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1, opt=None):
    if opt is not None and opt.noise_norm:
        n = np.prod(size)*num_samp
        values = np.random.rand(n)
        value_bins = np.searchsorted(opt.cdf, values)
        noise = opt.x_grid[value_bins]
        noise = torch.from_numpy(noise / 255.).type(torch.FloatTensor).to(device)
        noise = norm(noise).view([num_samp]+size)
        # print(noise.shape)
        # exit()
        # print(torch.min(noise), torch.max(noise), torch.mean(noise), torch.std(noise))
        # print(noise.shape)
        # exit()
    elif type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise,size[1], size[2])
    elif type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    elif type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    elif type == "truncated_normal":
        # noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = truncated_normal([num_samp, size[0], size[1], size[2]])
        # print(torch.min(noise), torch.max(noise), torch.mean(noise), torch.std(noise))
        # exit()
    return noise

def plot_learning_curves(G_loss,D_loss,epochs,label1,label2,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,G_loss,n,D_loss)
    #plt.title('loss')
    #plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend([label1,label2],loc='upper right')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def plot_learning_curve(loss,epochs,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,loss)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)

def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    MSGGan = False
    if  MSGGan:
        alpha = torch.rand(1, 1)
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = [alpha * rd + ((1 - alpha) * fd) for rd, fd in zip(real_data, fake_data)]
        interpolates = [i.to(device) for i in interpolates]
        interpolates = [torch.autograd.Variable(i, requires_grad=True) for i in interpolates]

        disc_interpolates = netD(interpolates)
    else:
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)#.cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates, _ = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def read_image(opt):
    if opt.train_multiple_images:
        x = []
        min_w = 100000
        min_h = 0
        for image in opt.imgs:
            _x = img.imread(image)
            _x = np2torch(_x,opt)
            _x = _x[:,0:3,:,:]
            if _x.shape[-1] < min_w:
                min_w = _x.shape[-1]
                min_h = _x.shape[-2]
            x.append(_x)
        for idx in range(len(x)):
            x[idx] = imresize_to_shape(x[idx], [min_h, min_w], opt)
    else:
        x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
        x = np2torch(x,opt)
        x = x[:,0:3,:,:]
    return x


def read_image_dir(dir,opt):
    x = img.imread('%s' % (dir))
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]
    return x


def np2torch_img_norm(opt, shape):
    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    if opt.nc_im == 3:
        x = x.transpose((2, 0, 1))
    else:
        raise NotImplementedError

    x = x[0:3, :, :]
    x = torch.from_numpy(x).type(torch.FloatTensor)
    x = torch.nn.functional.interpolate(torch.unsqueeze(x, 0), size=shape, mode='bilinear').squeeze()
    means = [torch.mean(x[idx]) for idx in range(x.shape[0])]
    stds = [torch.std(x[idx]) for idx in range(x.shape[0])]
    means = [134.6643, 134.6140, 66.1310]
    stds = [35.2716, 34.5341, 30.3217]
    opt.img_means = means
    opt.img_stds = stds
    x = TF.normalize(x, means, stds)
    x = torch.unsqueeze(x, 0)
    if not(opt.not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    return x


def np2torch(x,opt):
    if opt.nc_im == 3:
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255
    else:
        x = color.rgb2gray(x)
        x = x[:,:,None,None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not(opt.not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    #x = x.type(torch.FloatTensor)
    x = norm(x)
    return x

def torch2uint8(x):
    x = x[0,:,:,:]
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x

def read_image2np(opt):
    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    x = x[:, :, 0:3]
    return x

def save_networks(netG,netDs,z,opt, netDs_ratio=None):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    if isinstance(netDs, list):
        for i, netD in enumerate(netDs):
            torch.save(netD.state_dict(), '%s/netD_%s.pth' % (opt.outf, str(i)))
    else:
        torch.save(netDs.state_dict(), '%s/netD.pth' % (opt.outf))
    if opt.change_ratio and opt.num_Ds > 0:
        for i, ratio_D in enumerate(netDs_ratio):
            torch.save(ratio_D.state_dict(), '%s/netD_ratio_%s.pth' % (opt.outf, str(i)))
    torch.save(z, '%s/z_opt.pth' % (opt.outf))

def adjust_scales2image(real_,opt):
    if opt.train_multiple_images:
        real = real_
        real_ = real_[0]
    #opt.num_scales = int((math.log(math.pow(opt.min_size / (real_.shape[2]), 1), opt.scale_factor_init))) + 1
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real_.shape[2], real_.shape[3])), 1), opt.scale_factor_init))) + 1
    if opt.num_training_scales > 0:
        opt.num_scales = opt.num_training_scales
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]),1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    if opt.train_multiple_images:
        for idx in range(len(real)):
            real[idx] = imresize(real[idx], opt.scale1, opt)
        opt.scale_factor = math.pow(opt.min_size / (min(real[0].shape[2], real[0].shape[3])), 1 / (opt.stop_scale))
    else:
        real = imresize(real_, opt.scale1, opt)
        # print(real.shape)
        # print(opt.stop_scale)
        opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3])),1/(opt.stop_scale))
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def adjust_scales2image_SR(real_,opt):
    opt.min_size = 18
    opt.num_scales = int((math.log(opt.min_size / min(real_.shape[2], real_.shape[3]), opt.scale_factor_init))) + 1
    scale2stop = int(math.log(min(opt.max_size , max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1, opt)
    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3])),1/(opt.stop_scale))
    scale2stop = int(math.log(min(opt.max_size, max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def creat_reals_pyramid(real,reals,opt):
    # real = real[:,0:3,:,:]
    if opt.hc_scales:
        if opt.norm_img_scale:
            for i in range(0, opt.stop_scale + 1, 1):
                if opt.stop_scale - i == 0:
                    scale = 1.0
                else:
                    scale = math.pow(opt.scale_factor, ((opt.stop_scale - 1) / math.log(opt.stop_scale)) * math.log(
                        opt.stop_scale - i) + 1)
                image = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
                image = image[:, :, :3]
                image = imresize_in(image, scale_factor=scale)
                x = image[:, :, :, None]
                x = x.transpose((3, 2, 0, 1))
                _min = x.min()
                _max = x.max()
                x = (x - _min) / (_max - _min)
                x = torch.from_numpy(x)
                if not (opt.not_cuda):
                    x = move_to_gpu(x)
                x = x.type(torch.cuda.FloatTensor) if not (opt.not_cuda) else x.type(torch.FloatTensor)
                x = norm(x)
                print(x.shape)
                print(i, torch.min(x), torch.max(x), torch.mean(x))
                print("--")
                plt.imsave('test_scale_{}.png'.format(i), convert_image_np(x, opt))

                reals.append(x)
        if opt.train_multiple_images:
            reals = []
            for idx in range(len(real)):
                _reals = []
                for i in range(0, opt.stop_scale + 1, 1):
                    if opt.stop_scale - i == 0:
                        scale = 1.0
                    else:
                        scale = math.pow(opt.scale_factor, ((opt.stop_scale - 1) / math.log(opt.stop_scale)) * math.log(
                            opt.stop_scale - i) + 1)
                    curr_real = imresize(real[idx], scale, opt)
                    _reals.append(curr_real)
                reals.append(_reals)
        else:
            for i in range(0,opt.stop_scale+1,1):
                if opt.stop_scale-i == 0:
                    scale = 1.0
                else:
                    scale = math.pow(opt.scale_factor,((opt.stop_scale-1)/math.log(opt.stop_scale))*math.log(opt.stop_scale-i)+1)
                curr_real = imresize(real,scale,opt)
                reals.append(curr_real)
    else:
        if opt.train_multiple_images:
            reals = []
            for idx in range(len(real)):
                _reals = []
                for i in range(0, opt.stop_scale + 1, 1):
                    scale = math.pow(opt.scale_factor, opt.stop_scale - i)
                    curr_real = imresize(real[idx], scale, opt)
                    _reals.append(curr_real)
                reals.append(_reals)
        else:
            for i in range(0,opt.stop_scale+1,1):
                scale = math.pow(opt.scale_factor,opt.stop_scale-i)
                curr_real = imresize(real,scale,opt)
                reals.append(curr_real)
    return reals


def load_trained_pyramid(opt, mode_='train'):
    #dir = 'TrainedModels/%s/scale_factor=%f' % (opt.input_name[:-4], opt.scale_factor_init)
    mode = opt.mode
    opt.mode = 'train'
    if (mode == 'animation_train') | (mode == 'SR_train') | (mode == 'paint_train'):
        opt.mode = mode
    dir = generate_dir2save(opt)
    # print(dir)
    # exit()
    if(os.path.exists(dir)):
        Gs = torch.load('%s/Gs.pth' % dir, map_location="cuda:{}".format(torch.cuda.current_device()))
        Zs = torch.load('%s/Zs.pth' % dir, map_location="cuda:{}".format(torch.cuda.current_device()))
        reals = torch.load('%s/reals.pth' % dir, map_location="cuda:{}".format(torch.cuda.current_device()))
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    else:
        print('no appropriate trained model exists: {}'.format(dir))
    opt.mode = mode
    return Gs,Zs,reals,NoiseAmp

def generate_in2coarsest(reals,scale_v,scale_h,opt):
    real = reals[opt.gen_start_scale]
    real_down = upsampling(real, scale_v * real.shape[2], scale_h * real.shape[3])
    if opt.gen_start_scale == 0:
        in_s = torch.full(real_down.shape, 0, device=opt.device)
    else: #if n!=0
        in_s = upsampling(real_down, real_down.shape[2], real_down.shape[3])
    return in_s

def generate_dir2save(opt, timestamp=True):
    dir2save = None
    if (opt.mode == 'train') | (opt.mode == 'SR_train'):
        dir2save = 'TrainedModels/%s/scale_factor=%f,alpha=%d' % (opt.input_name[:-4], opt.scale_factor_init,opt.alpha)
    elif (opt.mode == 'animation_train') :
        dir2save = 'TrainedModels/%s/scale_factor=%f_noise_padding' % (opt.input_name[:-4], opt.scale_factor_init)
    elif (opt.mode == 'paint_train') :
        dir2save = 'TrainedModels/%s/scale_factor=%f_paint/start_scale=%d' % (opt.input_name[:-4], opt.scale_factor_init,opt.paint_start_scale)
    elif opt.mode == 'random_samples':
        dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out,opt.input_name[:-4], opt.gen_start_scale)
    elif opt.mode == 'random_samples_arbitrary_sizes':
        dir2save = '%s/RandomSamples_ArbitrerySizes/%s/scale_v=%f_scale_h=%f' % (opt.out,opt.input_name[:-4], opt.scale_v, opt.scale_h)
    elif opt.mode == 'animation':
        dir2save = '%s/Animation/%s' % (opt.out, opt.input_name[:-4])
    elif opt.mode == 'SR':
        dir2save = '%s/SR/%s' % (opt.out, opt.sr_factor)
    elif opt.mode == 'harmonization':
        dir2save = '%s/Harmonization/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
    elif opt.mode == 'editing':
        dir2save = '%s/Editing/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
    elif opt.mode == 'paint2image':
        dir2save = '%s/Paint2image/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
        if opt.quantization_flag:
            dir2save = '%s_quantized' % dir2save
    if timestamp:
        dir2save += "_" + opt.timestamp
    if opt.location:
        dir2save += "_location_{}".format(opt.beta)
    if opt.addDs:
        dir2save += "_addDs"
    if opt.ProSinGAN:
        dir2save += "_ProSinGAN"
    if opt.MSGGan:
        dir2save += "_MSGGan"
    if opt.growing:
        dir2save += "_growing"
        dir2save += "_depth_{}_lr_scale_{}".format(opt.train_depth, opt.lr_scale)
    if opt.augment:
        dir2save += "_augment"
    if opt.adaptive:
        dir2save += "_adaptive_" + str(opt.adaptive)
    if opt.self_attention_G:
        dir2save += "_saG"
    if opt.self_attention_D:
        dir2save += "_saD"
    if opt.start_scale > 0:
        dir2save += "_start_scale_" + str(opt.start_scale)
    if opt.train_mode != "generation" and opt.train_mode != "original":
        dir2save += "_train_scales_" + str(opt.train_scales)
    if opt.hc_scales:
        dir2save += "_hc_scales"
    if opt.change_ratio:
        dir2save += "_ratio"
        dir2save += "_rDs_" + str(opt.num_Ds)
    if opt.noise_norm:
        dir2save += "_noisenorm"
    if opt.log_layer_change:
        dir2save += "_llc"
    if opt.batch_norm != 1:
        dir2save += "_noBN"
    # print(opt.batch_norm)
    dir2save += "_act_" + opt.activation
    if opt.train_mode == "editing":
        dir2save += "_" + str(opt.lrelu_alpha)
    if opt.train_multiple_images:
        dir2save += "_manyimgs"
    if opt.add_img:
        dir2save += "_addimg"

    return dir2save

def post_config(opt):
    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:{}".format(opt.gpu))
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    opt.timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
    opt.out_ = 'TrainedModels/%s/scale_factor=%f/' % (opt.input_name[:-4], opt.scale_factor)
    if opt.mode == 'SR':
        opt.alpha = 100

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt

def calc_init_scale(opt):
    in_scale = math.pow(1/2,1/3)
    iter_num = round(math.log(1 / opt.sr_factor, in_scale))
    in_scale = pow(opt.sr_factor, 1 / iter_num)
    return in_scale,iter_num

def quant(prev,device):
    arr = prev.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, random_state=0).fit(arr)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    x = centers[labels]
    x = torch.from_numpy(x)
    x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if () else x.type(torch.FloatTensor)
    #x = x.type(torch.FloatTensor.to(device))
    x = x.view(prev.shape)
    return x,centers

def quant2centers(paint, centers):
    arr = paint.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, init=centers, n_init=1).fit(arr)
    labels = kmeans.labels_
    #centers = kmeans.cluster_centers_
    x = centers[labels]
    x = torch.from_numpy(x)
    x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if torch.cuda.is_available() else x.type(torch.FloatTensor)
    #x = x.type(torch.cuda.FloatTensor)
    x = x.view(paint.shape)
    return x

    return paint


def dilate_mask(mask,opt):
    if opt.mode == "harmonization":
        element = morphology.disk(radius=7)
    if opt.mode == "editing":
        element = morphology.disk(radius=20)
    mask = torch2uint8(mask)
    mask = mask[:,:,0]
    mask = morphology.binary_dilation(mask,selem=element)
    mask = filters.gaussian(mask, sigma=5)
    nc_im = opt.nc_im
    opt.nc_im = 1
    mask = np2torch(mask,opt)
    opt.nc_im = nc_im
    mask = mask.expand(1, 3, mask.shape[2], mask.shape[3])
    plt.imsave('%s/%s_mask_dilated.png' % (opt.ref_dir, opt.ref_name[:-4]), convert_image_np(mask, opt), vmin=0,vmax=1)
    mask = (mask-mask.min())/(mask.max()-mask.min())
    return mask


def shuffle_grid(image, max_tiles=5):
    tiles = []
    img_w, img_h = image.shape[0], image.shape[1]
    _max_tiles = random.randint(1, max_tiles)
    # _max_tiles = random.randint(3,3)
    if _max_tiles == 1:
        w_min, h_min = int(img_w*0.2), int(img_h*0.2)
        w_max, h_max = int(img_w*0.5), int(img_h*0.5)
        x_translation_min, y_translation_min = int(img_w*0.05), int(img_h*0.05)
        x_translation_max, y_translation_max = int(img_w*0.15), int(img_h*0.15)
    elif _max_tiles == 2:
        w_min, h_min = int(img_w*0.15), int(img_h*0.15)
        w_max, h_max = int(img_w*0.3), int(img_h*0.3)
        x_translation_min, y_translation_min = int(img_w*0.05), int(img_h*0.05)
        x_translation_max, y_translation_max = int(img_w*0.1), int(img_h*0.1)
    elif _max_tiles == 3:
        w_min, h_min = int(img_w*0.1), int(img_h*0.1)
        w_max, h_max = int(img_w*0.2), int(img_h*0.2)
        x_translation_min, y_translation_min = int(img_w*0.05), int(img_h*0.05)
        x_translation_max, y_translation_max = int(img_w*0.1), int(img_h*0.1)
    else:
        w_min, h_min = int(img_w*0.1), int(img_h*0.1)
        w_max, h_max = int(img_w*0.15), int(img_h*0.15)
        x_translation_min, y_translation_min = int(img_w*0.05), int(img_h*0.05)
        x_translation_max, y_translation_max = int(img_w*0.1), int(img_h*0.1)

    for idx in range(_max_tiles):
        x, y = random.randint(0, img_w), random.randint(0, img_h)
        w, h = random.randint(w_min, w_max), random.randint(h_min, h_max)
        if x + w >= img_w:
            w = img_w - x
        if y + h >= img_h:
            h = img_h - y
        x_t, y_t = random.randint(x_translation_min, x_translation_max), random.randint(y_translation_min, y_translation_max)
        if random.random() < 0.5:
            x_t, y_t = -x_t, -y_t
            if x + x_t < 0:
                x_t = -x
            if y + y_t < 0:
                y_t = -y
        else:
            if x + x_t + w >= img_w:
                x_t = img_w - w - x
            if y + y_t + h >= img_h:
                y_t = img_h - h - y
        tiles.append([x, y, w, h, x+x_t, y+y_t])

    new_image = copy.deepcopy(image)
    for tile in tiles:
        x, y, w, h, x_new, y_new = tile
        new_image[x_new:x_new+w, y_new:y_new+h, :] = image[x:x+w, y:y+h, :]

    return new_image


class Augment():
    def __init__(self, img_h=None, img_w=None):
        super().__init__()
        self._transofrm = self.strong_aug()

    def strong_aug(self):
        color_r = random.randint(0, 256)
        color_g = random.randint(0, 256)
        color_b = random.randint(0, 256)
        num_holes = random.randint(1, 2)
        if num_holes == 2:
            max_h_size = random.randint(15, 30)
            max_w_size = random.randint(15, 30)
        else:
            max_h_size = random.randint(30, 60)
            max_w_size = random.randint(30, 60)
        return Compose([
            OneOf([
                OneOf([
                    MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=0.2),
                    IAAAdditiveGaussianNoise(),
                    GaussNoise()]),
                OneOf([
                    InvertImg(),
                    ToSepia()]),
                OneOf([
                    ChannelDropout(channel_drop_range=(1, 1), fill_value=0),
                    ChannelShuffle()]),
                HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.1)],
                p=0.25),
            Cutout(num_holes=num_holes, max_h_size=max_h_size, max_w_size=max_w_size,
                   fill_value=[color_r, color_g, color_b], p=0.9),
            # OneOf([
            #     Cutout(num_holes=5, max_h_size=40, max_w_size=40, fill_value=[color_r,color_g,color_b]),
            #     Cutout(num_holes=5, max_h_size=40, max_w_size=40, fill_value=[color_r,color_g,color_b]),
            #     Cutout(num_holes=5, max_h_size=40, max_w_size=40, fill_value=[color_r,color_g,color_b]),
            #     Cutout(num_holes=5, max_h_size=40, max_w_size=40, fill_value=[color_r,color_g,color_b])],
            #     p=0.9),
            #     RandomRotate90(),
            #     Flip(),
            #     Transpose(),
            #     OneOf([
            #         IAAAdditiveGaussianNoise(),
            #         GaussNoise(),
            #     ], p=0.8),
            #     OneOf([
            #         MotionBlur(p=0.2),
            #         MedianBlur(blur_limit=3, p=0.1),
            #         Blur(blur_limit=3, p=0.1),
            #     ], p=0.8),
            #     ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            #     OneOf([
            #         OpticalDistortion(p=0.3),
            #         GridDistortion(p=0.1),
            #         IAAPiecewiseAffine(p=0.3),
            #     ], p=0.8),
            #     OneOf([
            #         CLAHE(clip_limit=2),
            #         IAASharpen(),
            #         IAAEmboss(),
            #         RandomBrightnessContrast(),
            #     ], p=0.8),
            #     HueSaturationValue(p=0.8),
            # ], p=p
        ])

    def transform(self, **x):
        _transform = self.strong_aug()
        return _transform(**x)

# class Augment():
#     def __init__(self, img_h, img_w):
#         super().__init__()
#         self.img_h = img_h
#         self.img_w = img_w
#
#     def strong_aug(self, **kwargs):
#         color_r = random.randint(0, 256)
#         color_g = random.randint(0, 256)
#         color_b = random.randint(0, 256)
#         num_holes = random.randint(1, 2)
#         if num_holes == 2:
#             min_w, max_w = int(0.1*self.img_w), int(0.25*self.img_w)
#             min_h, max_h = int(0.1*self.img_w), int(0.25*self.img_w)
#         else:
#             min_w, max_w = int(0.2*self.img_w), int(0.5*self.img_w)
#             min_h, max_h = int(0.2*self.img_w), int(0.5*self.img_w)
#         max_h_size = random.randint(min_h, max_h)
#         max_w_size = random.randint(min_w, max_w)
#         return Compose([
#             OneOf([
#                 OneOf([
#                     MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True),
#                     IAAAdditiveGaussianNoise(),
#                     GaussNoise()]),
#                 OneOf([
#                     InvertImg(),
#                     ToSepia()]),
#                 OneOf([
#                     ChannelDropout(channel_drop_range=(1, 1), fill_value=0),
#                     ChannelShuffle()]),
#                 HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20)],
#                 p=0.25),
#             Cutout(num_holes=num_holes, max_h_size=max_h_size, max_w_size=max_w_size,
#                    fill_value=[color_r, color_g, color_b], p=0.9),
#             # OneOf([
#             #     Cutout(num_holes=5, max_h_size=40, max_w_size=40, fill_value=[color_r,color_g,color_b]),
#             #     Cutout(num_holes=5, max_h_size=40, max_w_size=40, fill_value=[color_r,color_g,color_b]),
#             #     Cutout(num_holes=5, max_h_size=40, max_w_size=40, fill_value=[color_r,color_g,color_b]),
#             #     Cutout(num_holes=5, max_h_size=40, max_w_size=40, fill_value=[color_r,color_g,color_b])],
#             #     p=0.9),
#             #     RandomRotate90(),
#             #     Flip(),
#             #     Transpose(),
#             #     OneOf([
#             #         IAAAdditiveGaussianNoise(),
#             #         GaussNoise(),
#             #     ], p=0.8),
#             #     OneOf([
#             #         MotionBlur(p=0.2),
#             #         MedianBlur(blur_limit=3, p=0.1),
#             #         Blur(blur_limit=3, p=0.1),
#             #     ], p=0.8),
#             #     ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
#             #     OneOf([
#             #         OpticalDistortion(p=0.3),
#             #         GridDistortion(p=0.1),
#             #         IAAPiecewiseAffine(p=0.3),
#             #     ], p=0.8),
#             #     OneOf([
#             #         CLAHE(clip_limit=2),
#             #         IAASharpen(),
#             #         IAAEmboss(),
#             #         RandomBrightnessContrast(),
#             #     ], p=0.8),
#             #     HueSaturationValue(p=0.8),
#             # ], p=p
#         ])
#
#     def transform(self, **x):
#         _transform = self.strong_aug()
#         return _transform(**x)



class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self.lr_updates = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        self.lr_updates += 1
        if self.verbose:
            print("Updating LR")
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                # if self.verbose:
                #     print('Epoch {:5d}: reducing learning rate'
                #           ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)