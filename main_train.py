import datetime
import dateutil.tz
import os
import os.path as osp
from shutil import copyfile, copytree
import glob
import time
import random
import torch

from ConSinGAN.config import get_arguments
import ConSinGAN.functions as functions


def get_scale_factor(opt):
    opt.scale_factor = 1.0
    num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real.shape[2], real.shape[3])), 1), opt.scale_factor_init))) + 1
    opt.scale_factor_init = opt.scale_factor
    if opt.num_training_scales > 0:
        while num_scales > opt.num_training_scales:
            opt.scale_factor_init = opt.scale_factor_init - 0.01
            num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real.shape[2], real.shape[3])), 1), opt.scale_factor_init))) + 1
    return opt.scale_factor_init


# noinspection PyInterpreter
if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--gpu', type=int, help='which GPU', default=0)
    parser.add_argument('--train_stages', type=int, help='which GPU', default=6)

    parser.add_argument('--sample', action='store_true', help='generate random samples', default=0)
    parser.add_argument('--train_depth', type=int, help='how many layers are trained if growing', default=3)
    parser.add_argument('--lr_scale', type=float, help='scaling of learning rate for layers if growing', default=0.1)
    parser.add_argument('--start_scale', type=int, help='at which scale to start training', default=0)
    parser.add_argument('--train_scales', type=int, help='at which scale to start training', default=3)
    parser.add_argument('--harmonization_img', help='for harmonization', type=str, default='')
    parser.add_argument('--add_img', action='store_true', help='use augmented img for adversarial loss', default=0)
    parser.add_argument('--fine_tune', action='store_true', help='fine tune on final image', default=0)
    parser.add_argument('--fine_tune_model', action='store_true', help='fine tune on final image', default=0)
    parser.add_argument('--model_finetune_dir', help='input image name', required=False)
    parser.add_argument('--hq', action='store_true', help='fine tune on high res image', default=0)
    parser.add_argument('--add_mask', action='store_true', help='fine tune on high res image', default=0)
    parser.add_argument('--num_training_scales', type=int, help='how many scales to train on', default=0)
    parser.add_argument('--edit_add_noise', action='store_true', help='fine tune on high res image', default=0)

    parser.add_argument('--batch_norm', action='store_true', help='"use batch norm in generator"', default=0)


    opt = parser.parse_args()
    opt = functions.post_config(opt)

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)

    if opt.train_mode == "generation":
        from ConSinGAN.training_generation import *
    elif opt.train_mode == "retarget":
        from ConSinGAN.training_retarget import *
    elif opt.train_mode == "harmonization":
        if opt.fine_tune_model:
            if opt.hq:
                from ConSinGAN.training_prosingan_harmonization_finetune_model_highres import *
            else:
                from ConSinGAN.training_prosingan_harmonization_finetune_model import *
        else:
            from ConSinGAN.training_prosingan_harmonization import *
    elif opt.train_mode == "editing":
        if opt.fine_tune_model:
            from ConSinGAN.training_prosingan_editing_finetune_model import *
        else:
            from ConSinGAN.training_prosingan_editing import *


    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []

    real = functions.read_image(opt)
    dir2save = functions.generate_dir2save(opt)

    if osp.exists(dir2save):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        with open(osp.join(dir2save, 'opt.txt'), 'w') as f:
            for o in opt.__dict__:
                f.write("{}\t-\t{}\n".format(o, opt.__dict__[o]))
        current_path = os.path.dirname(os.path.abspath(__file__))
        for py_file in glob.glob(osp.join(current_path, "*.py")):
            copyfile(py_file, osp.join(dir2save, py_file.split("/")[-1]))
        copytree(osp.join(current_path, "ConSinGAN"), osp.join(dir2save, "ConSinGAN"))

        print("Training model ({})".format(opt.timestamp))
        functions.adjust_scales2image(real, opt)

        start = time.time()
        train(opt, Gs, Zs, reals, NoiseAmp)
        end = time.time()
        elapsed_time = end - start
        print("Time for training: {} seconds".format(elapsed_time))
