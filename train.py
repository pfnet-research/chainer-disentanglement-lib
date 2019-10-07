import os
import random
import argparse

import numpy as np
import chainer
from chainer.training.updaters import StandardUpdater

import utils_chainer as u_chain
from net.shared import make_optimizer, train

parser = argparse.ArgumentParser(description='train VAEs')
parser.add_argument('--vae', type=str, default="BetaVAE", metavar='N',
                    help='set VAE type to train')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--training_steps', type=int, default=300000, metavar='N',
                    help='number iterations (default: 300000)')
parser.add_argument('--device', type=int, default=0,
                    help='GPU device num, set -1 for CPU training')
parser.add_argument('--n_z', type=int, default=10, metavar='S',
                    help='latent dimension (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--flat', type=int, default=0, metavar='S',
                    help='flat input data (default: 0)')
args = parser.parse_args()

# set GPU
if chainer.cuda.available:
    import cupy as cp
    GPU = args.device
else:
    GPU = -1

# path to use
base_path = os.getenv("OUTPUT_PATH", "../results")
experiment_name = os.getenv("EVALUATION_NAME", "dev_tmp")
out_path = os.path.join(base_path, experiment_name)

# for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
if chainer.cuda.available:
    cp.random.seed(args.seed)
os.environ["CHAINER_SEED"] = str(args.seed)
chainer.global_config.cudnn_deterministic = True

# load dataset
train_loader = u_chain.get_loader(batch_size=args.batch_size, flat=args.flat)
channel_num = train_loader.dataset.get_example(0).shape[0]

# load models
with_dis = False
if args.vae == "BetaVAE":
    from net.betavae import BetaVAE
    vae = BetaVAE(n_z=args.n_z, channel_num=channel_num)
    optimizer = make_optimizer(vae, 0.0001, 0.9, 0.999)
    updater = StandardUpdater(train_loader, optimizer,
                              device=GPU, loss_func=vae.get_loss_func())
    log_reports = ['iteration', 'main/loss', 'main/rec_loss',
                   'main/lat_loss', 'elapsed_time']
    pass
elif args.vae == "FactorVAE":
    from net.factorvae import FactorVAE, Discriminator, FactorUpdater
    vae = FactorVAE(n_z=args.n_z, channel_num=channel_num)
    discriminator = Discriminator(n_z=args.n_z)
    models = [vae, discriminator]

    opts = {
        "opt_vae": make_optimizer(vae, 0.0001, 0.9, 0.999),
        "opt_dis": make_optimizer(discriminator, 0.0001, 0.5, 0.9)
    }

    updater_args = {
        "iterator": {'main': train_loader},
        "device": GPU,
        "optimizer": opts,
        "models": models
    }
    updater = FactorUpdater(**updater_args)
    log_reports = ['iteration', 'vae_loss', 'rec_loss',
                   'lat_loss', 'tc_loss', 'dis_loss', 'elapsed_time']
    with_dis = True
    pass
elif args.vae == "DIPVAE-1":
    from net.dipvae import DIPVAE
    vae = DIPVAE(n_z=args.n_z, dip_type="i", lambda_od=50,
                 lambda_d=500, channel_num=channel_num)
    optimizer = make_optimizer(vae, 0.0001, 0.9, 0.999)
    updater = StandardUpdater(train_loader, optimizer,
                              device=GPU, loss_func=vae.get_loss_func())
    log_reports = ['iteration', 'main/loss',
                   'main/rec_loss', 'main/lat_loss',
                   'main/dip_loss', 'elapsed_time']
    pass
elif args.vae == "DIPVAE-2":
    from net.dipvae import DIPVAE
    vae = DIPVAE(n_z=args.n_z, dip_type="ii", lambda_od=50,
                 lambda_d=50, channel_num=channel_num)
    optimizer = make_optimizer(vae, 0.0001, 0.9, 0.999)
    updater = StandardUpdater(train_loader, optimizer,
                              device=GPU, loss_func=vae.get_loss_func())
    log_reports = ['iteration', 'main/loss',
                   'main/rec_loss', 'main/lat_loss',
                   'main/dip_loss', 'elapsed_time']
    pass
elif args.vae == "JointVAE":
    from net.jointvae import JointVAE, JointUpdater
    vae = JointVAE(channel_num=channel_num)
    opts = {
        "opt_vae": make_optimizer(vae, 0.0001, 0.9, 0.999),
    }

    updater_args = {
        "iterator": {'main': train_loader},
        "device": GPU,
        "optimizer": opts,
        "model": vae
    }
    updater = JointUpdater(**updater_args)
    log_reports = ['iteration', 'vae_loss', 'rec_loss',
                   'cont_loss', 'disc_loss']
else:
    raise NotImplementedError


if __name__ == '__main__':
    # Train
    print("---- start training ----")
    train(updater, args.training_steps, log_reports, out_path, with_dis)
    print("---- finish training ----")

    # Export the representation extractor
    vae.to_cpu()
    u_chain.export_model(vae)

    # save results
    print("---- make reconstruction img ----")
    x = chainer.Variable(np.asarray(train_loader.next()[:9]))
    u_chain.compare_rec_images(vae, x, "rec_compare")
    if args.vae != "JointVAE":
        # show_latent_gif not support JointVAE
        print("---- make latent traversal ----")
        for i in range(2):
            x = chainer.Variable(np.asarray(train_loader.next()[i:i + 1]))
            u_chain.show_latent_gif(vae, x, "latent_traversal_" + str(i))
