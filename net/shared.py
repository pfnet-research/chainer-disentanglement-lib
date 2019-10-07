import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer.functions.loss.vae import gaussian_kl_divergence


class BaseVAE(chainer.Chain):
    """Base VAE model."""
    def __init__(self, n_z=10, beta=1, k=1, channel_num=3):
        """
        Args:
            n_z (int) :  latent dimension
            beta (int) : Regularization strength for KL divergence.
            k (int) : Number of Monte Carlo samples used in encoded vector.
        """
        super(BaseVAE, self).__init__()
        with self.init_scope():
            # Prameters
            self.beta = beta
            self.k = k
            self.args = {"n_z": n_z, "beta": beta,
                         "k": k, "channel_num": channel_num}

            initializer = chainer.initializers.GlorotUniform()
            # encoder
            self.conv1 = L.Convolution2D(in_channels=channel_num, out_channels=32,
                                         ksize=4, stride=2, pad=1, initialW=initializer)
            self.conv2 = L.Convolution2D(in_channels=32, out_channels=32,
                                         ksize=4, stride=2, pad=1, initialW=initializer)
            self.conv3 = L.Convolution2D(in_channels=32, out_channels=64,
                                         ksize=4, stride=2, pad=1, initialW=initializer)
            self.conv4 = L.Convolution2D(in_channels=64, out_channels=64,
                                         ksize=4, stride=2, pad=1, initialW=initializer)
            self.fc1 = L.Linear(1024, 256, initialW=initializer)
            self.fc2_mu = L.Linear(256, n_z, initialW=initializer)
            self.fc2_ln_var = L.Linear(256, n_z, initialW=initializer)
            # decoder
            self.fc3 = L.Linear(n_z, 256, initialW=initializer)
            self.fc4 = L.Linear(256, 1024, initialW=initializer)
            self.deconv1 = L.Deconvolution2D(in_channels=64, out_channels=64,
                                             ksize=4, stride=2, pad=1, initialW=initializer)
            self.deconv2 = L.Deconvolution2D(in_channels=64, out_channels=32,
                                             ksize=4, stride=2, pad=1, initialW=initializer)
            self.deconv3 = L.Deconvolution2D(in_channels=32, out_channels=32,
                                             ksize=4, stride=2, pad=1, initialW=initializer)
            self.deconv4 = L.Deconvolution2D(in_channels=32, out_channels=channel_num,
                                             ksize=4, stride=2, pad=1, initialW=initializer)

    def forward(self, x, sigmoid=True, mode="mean", represent=False):
        mu, ln_var = self.encode(x)
        if mode == "mean":
            z = mu
        elif mode == "sample":
            z = F.gaussian(mu, ln_var)
        else:
            raise NotImplementedError

        if represent:
            return z
        else:
            return self.decode(z, sigmoid=sigmoid)

    def encode(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.fc1(h))
        mu = self.fc2_mu(h)
        ln_var = self.fc2_ln_var(h)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        h = F.reshape(h, (-1, 64, 4, 4))
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        h = self.deconv4(h)
        if sigmoid:
            return F.sigmoid(h)
        else:
            return h

    def get_loss_func(self):
        def lf(x):
            # x = x.reshape(-1, 3*64*64)
            mu, ln_var = self.encode(x)
            batchsize = len(mu.data)

            # reconstruction loss
            rec_loss = 0
            for _ in range(self.k):
                z = F.gaussian(mu, ln_var)
                rec_loss += F.bernoulli_nll(x, self.decode(z, sigmoid=False)) \
                    / (self.k * batchsize)
            self.rec_loss = rec_loss

            # latent loss
            lat_loss = self.beta * gaussian_kl_divergence(mu, ln_var) / batchsize
            self.lat_loss = lat_loss

            self.loss = rec_loss + lat_loss
            chainer.report({"rec_loss": rec_loss, "lat_loss": lat_loss, "loss": self.loss},
                           observer=self)
            return self.loss
        return lf


def make_optimizer(model, alpha, beta1, beta2):
    """
    make Adam optimizer
    Parameters
    ----------
    updater : chainer.Chain
        updater to train
    alpha : float
        Adam's hyperparameter alpha
    beta1 : float
        Adam's hyperparameter beta1
    beta2 : float
        Adam's hyperparameter beta2
    -------
    Adam optimizer
    """
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer


def train(updater, training_steps, log_reports, out_path, with_dis=False):
    """
    Training VAE
    Parameters
    ----------
    updater : chainer.training.StandardUpdater
        updater with VAE
    training_steps : iter
        number of iteration
    log_reports : [str, str, ...]
       losses to record
       second item should be "main/loss" or "vae_loss"
       third item should be "main/rec_loss" or "rec_loss"
    out_path : str
        path to save model
    with_dis : bool
        if your vae has discriminator, plot learning curve for it
        Ex. FactorVAE
    Returns
    -------
    None
    """
    trainer = training.Trainer(updater, (training_steps, "iteration"),
                               out=out_path)
    trainer.extend(extensions.dump_graph(log_reports[1]))
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))
    trainer.extend(extensions.PrintReport(log_reports))
    trainer.extend(extensions.PlotReport([log_reports[1], log_reports[2]],
                                         'iteration', file_name='loss.png',
                                         trigger=(100, 'iteration')))
    if with_dis:
        trainer.extend(extensions.PlotReport(['dis_loss'], 'iteration',
                                             file_name='dis_loss.png',
                                             trigger=(100, 'iteration')))
    trainer.extend(extensions.ProgressBar(update_interval=100))
    trainer.run()
    return
