import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence

from net.shared import BaseVAE


class DIPVAE(BaseVAE):
    """DIP-VAE model.
        Based on Equation (6) and (7) of "Variational Inference of Disentangled Latent
        Concepts from Unlabeled Observations"
        (https://openreview.net/pdf?id=H1kG7GZAW).
    """
    def __init__(self, n_z=10, k=1, channel_num=3,
                 dip_type="i", lambda_od=10, lambda_d=100):
        """
        Args:
            lambda_od : Hyperparameter for off diagonal values of covariance matrix.
            lambda_d   : Hyperparameter for diagonal values of covariance matrix
            dip_type: "i" or "ii".
        """
        super(DIPVAE, self).__init__(n_z=n_z, k=k, channel_num=channel_num)
        with self.init_scope():
            self.lambda_od = lambda_od
            self.lambda_d = lambda_d
            if dip_type != "i" and dip_type != "ii":
                raise NotImplementedError("DIP variant not supported.")
            self.dip_type = dip_type
            self.args = {"n_z": n_z, "k": k, "channel_num": channel_num,
                         "dip_type": dip_type, "lambda_od": lambda_od,
                         "lambda_d": lambda_d}

    def get_loss_func(self):
        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu.data)

            # reconstruction loss
            rec_loss = 0
            for _ in range(self.k):
                z_sampled = F.gaussian(mu, ln_var)
                rec_logits = self.decode(z_sampled, sigmoid=False)
                rec_loss += F.bernoulli_nll(x, rec_logits) / (self.k * batchsize)
            self.rec_loss = rec_loss

            # latent loss
            lat_loss = self.beta * gaussian_kl_divergence(mu, ln_var) / batchsize
            self.lat_loss = lat_loss

            # dip loss
            dip_loss = self.regularizer(mu, ln_var, z_sampled)

            self.loss = rec_loss + lat_loss + dip_loss
            chainer.report({"rec_loss": rec_loss, "lat_loss": lat_loss,
                            "dip_loss": dip_loss, "loss": self.loss},
                           observer=self)
            return self.loss
        return lf

    def regularizer(self, mu, ln_var, z_sampled):
        xp = mu.xp
        cov_mu = compute_covariance_mu(mu)
        if self.dip_type == "i":
            cov_dip_regularizer = regularize_diag_off_diag_dip(
                cov_mu, self.lambda_od, self.lambda_d)

        elif self.dip_type == "ii":
            cov_enc = F.expand_dims(F.exp(ln_var), 2) * xp.eye(mu.shape[1])
            e_cov_enc = F.mean(cov_enc, axis=0)
            cov_z = e_cov_enc + cov_mu
            cov_dip_regularizer = regularize_diag_off_diag_dip(
                cov_z, self.lambda_od, self.lambda_d)
        return cov_dip_regularizer


def compute_covariance_mu(mu):
    """Computes the covariance of mu.
    Uses cov(mu) = E[mumu^T] - E[mu]E[mu]^T.
    Args:
    mu: [batch_size, num_latent]
        Encoder's mean
    Returns:
    cov_mu: [num_latent, num_latent]
        Covariance of encoder mean
    """
    e_mu_mu_t = F.mean(F.expand_dims(mu, 2) * F.expand_dims(mu, 1), axis=0)
    e_mu = F.mean(mu, axis=0)
    e_mu_e_mu_T = F.expand_dims(e_mu, axis=1) * F.expand_dims(e_mu, axis=0)
    cov_mu = e_mu_mu_t - e_mu_e_mu_T
    return cov_mu


def regularize_diag_off_diag_dip(cov_mu, lambda_od, lambda_d):
    """Compute on and off diagonal regularizers for DIP-VAE models.
    Penalize deviations of covariance_matrix from the identity matrix. Uses
    different weights for the deviations of the diagonal and off diagonal entries.
    Args:
    cov_mu : [num_latent, num_latent]
        to regularize.
    lambda_od: Weight of penalty for off diagonal elements.
    lambda_d: Weight of penalty for diagonal elements.
    Returns:
    dip_regularizer: Regularized deviation from diagonal of covariance_matrix.
    """
    xp = cov_mu.xp
    cov_mu_diag = F.diagonal(cov_mu)
    cov_mu_off_diag = cov_mu - cov_mu_diag * xp.eye(cov_mu.shape[0])
    dip_regularizer = lambda_od * F.sum(cov_mu_off_diag ** 2) \
        + lambda_d * F.sum((cov_mu_diag - 1) ** 2)
    return dip_regularizer
