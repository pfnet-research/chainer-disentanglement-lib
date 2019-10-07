import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import distributions as D
from chainer.functions.loss.vae import gaussian_kl_divergence

from net.shared import BaseVAE


class FactorVAE(BaseVAE):
    """FactorVAE model.
    Based on Equation (2) of "Disentangling by Factorising"
    (https://arxiv.org/pdf/1802.05983.pdf)
    """
    def __init__(self, n_z=10, gamma=100, k=1, channel_num=3):
        """
        Args:
            n_z (int) : Latent dimension
            gamma (int) : Regularization strength for TC loss
            k (int) : Number of Monte Carlo samples used in encoded vector.
        """
        super(FactorVAE, self).__init__(n_z=n_z, k=k,
                                        channel_num=channel_num)
        with self.init_scope():
            self.gamma = gamma
            self.args = {"n_z": n_z, "gamma": gamma,
                         "k": k, "channel_num": channel_num}


class Discriminator(chainer.Chain):
    def __init__(self, n_z=10):
        """
        Args:
            n_z (int) : Latent dimension
        """
        super(Discriminator, self).__init__()
        with self.init_scope():
            initializer = chainer.initializers.GlorotUniform()
            self.fc1 = L.Linear(n_z, 1000, initialW=initializer)
            self.fc2 = L.Linear(1000, 1000, initialW=initializer)
            self.fc3 = L.Linear(1000, 1000, initialW=initializer)
            self.fc4 = L.Linear(1000, 1000, initialW=initializer)
            self.fc5 = L.Linear(1000, 1000, initialW=initializer)
            self.fc6 = L.Linear(1000, 1000, initialW=initializer)
            self.fc7 = L.Linear(1000, 2, initialW=initializer)

    def forward(self, z):
        h = F.leaky_relu(self.fc1(z))
        h = F.leaky_relu(self.fc2(h))
        h = F.leaky_relu(self.fc3(h))
        h = F.leaky_relu(self.fc4(h))
        h = F.leaky_relu(self.fc5(h))
        h = F.leaky_relu(self.fc6(h))
        logits = self.fc7(h)
        probs = F.softmax(logits)
        probs = F.clip(probs, 1e-6, 1 - 1e-6)
        return logits, probs


class FactorUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.vae, self.dis = kwargs.pop('models')
        super(FactorUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        vae_optimizer = self.get_optimizer('opt_vae')
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = self.vae.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x = chainer.dataset.concat_examples(batch, device=self.device)

        mu, ln_var = self.vae.encode(x)
        z_sampled = F.gaussian(mu, ln_var)
        z_shuffled = shuffle_codes(z_sampled)

        logits_z, probs_z = self.dis(z_sampled)
        _, probs_z_shuffle = self.dis(z_shuffled)

        reconstructions = self.vae.decode(z_sampled, sigmoid=False)

        # reconstruction loss
        rec_loss = 0
        for _ in range(self.vae.k):
            rec_loss += F.bernoulli_nll(x, reconstructions) \
                / (self.vae.k * batchsize)
        # latent loss
        lat_loss = self.vae.beta * gaussian_kl_divergence(mu, ln_var) / batchsize

        tc_loss = F.mean(logits_z[:, 0] - logits_z[:, 1])
        factor_vae_loss = rec_loss + lat_loss + self.vae.gamma * tc_loss
        dis_loss = -(0.5 * F.mean(F.log(probs_z[:, 0])) \
                     + 0.5 * F.mean(F.log(probs_z_shuffle[:, 1])))

        self.vae.cleargrads()
        self.dis.cleargrads()
        factor_vae_loss.backward()
        vae_optimizer.update()

        # avoid backword duplicate
        z_sampled.unchain_backward()
        self.dis.cleargrads()
        self.vae.cleargrads()
        dis_loss.backward()
        dis_optimizer.update()

        chainer.reporter.report({"rec_loss": rec_loss, "lat_loss": lat_loss,
                                 "tc_loss": tc_loss, "vae_loss": factor_vae_loss,
                                 "dis_loss": dis_loss})
        return


def shuffle_codes(z_sampled):
    """Shuffles latent variables across the batch.
        Args:
            z: [batch_size, num_latent]
                encoded representation.
        Returns:
            shuffled: [batch_size, num_latent]
                shuffled representation across the batch.
    """
    xp = z_sampled.xp
    z_shuffled = xp.array(z_sampled.data)
    for i in range(z_sampled.shape[1]):
        xp.random.shuffle(z_shuffled[:, i])
    return z_shuffled
