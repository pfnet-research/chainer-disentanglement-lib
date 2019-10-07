import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence


class JointVAE(chainer.Chain):
    """JointVAE model.
    Based on Equation (7) of "Learning Disentangled Joint Continuous
    and Discrete Representations"
    (https://arxiv.org/pdf/1804.00104.pdf)
    """
    def __init__(self, latent_spec={"cont": 6, "disc": [3]},
                 cont_capacity=[0.0, 40, 300000, 150],
                 disc_capacity=[0.0, 1.1, 300000, 150],
                 gumbel_temperature=0.67, k=1, channel_num=3):
        """
        Args:
            latent_spec : dict
                Specifies latent distribution. For example:
                {'cont': 10, 'disc': [10, 4, 3]} encodes 10 normal variables and
                3 gumbel softmax variables of dimension 10, 4 and 3. A latent spec
                can include both 'cont' and 'disc' or only 'cont' or only 'disc'.

        cont_capacity : tuple (float, float, int, float) or None
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_z).
            Parameters to control the capacity of the continuous latent
            channels. Cannot be None if model.is_continuous is True.

        disc_capacity : tuple (float, float, int, float) or None
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_c).
            Parameters to control the capacity of the discrete latent channels.
            Cannot be None if model.is_discrete is True.

        k (int): Number of Monte Carlo samples used in encoded vector.
        """
        super(JointVAE, self).__init__()
        with self.init_scope():
            # Prameters
            self.is_continuous = 'cont' in latent_spec
            self.is_discrete = 'disc' in latent_spec
            self.latent_spec = latent_spec
            self.temperature = gumbel_temperature
            self.cont_capacity = cont_capacity
            self.disc_capacity = disc_capacity
            self.k = k
            self.args = {"latent_spec": latent_spec, "cont_capacity": cont_capacity,
                         "disc_capacity": disc_capacity,
                         "gumbel_temperature": gumbel_temperature,
                         "k": k, "channel_num": channel_num}

            initializer = chainer.initializers.GlorotUniform()

            if self.is_continuous and self.cont_capacity is None:
                raise RuntimeError("Model has continuous but cont_capacity not provided.")

            if self.is_discrete and self.disc_capacity is None:
                raise RuntimeError("Model has discrete but disc_capacity not provided.")

            # Calculate dimensions of latent
            self.latent_cont_dim = 0
            self.latent_disc_dim = 0
            self.num_disc_latents = 0
            if self.is_continuous:
                self.latent_cont_dim = self.latent_spec['cont']
            if self.is_discrete:
                self.latent_disc_dim = sum([dim for dim in self.latent_spec['disc']])
                self.num_disc_latents = len(self.latent_spec['disc'])
            self.latent_dim = self.latent_cont_dim + self.latent_disc_dim

            # Encoder
            self.conv1 = L.Convolution2D(in_channels=channel_num, out_channels=32,
                                         ksize=4, stride=2, pad=1, initialW=initializer)
            self.conv2 = L.Convolution2D(in_channels=32, out_channels=32,
                                         ksize=4, stride=2, pad=1, initialW=initializer)
            self.conv3 = L.Convolution2D(in_channels=32, out_channels=64,
                                         ksize=4, stride=2, pad=1, initialW=initializer)
            self.conv4 = L.Convolution2D(in_channels=64, out_channels=64,
                                         ksize=4, stride=2, pad=1, initialW=initializer)
            self.fc1 = L.Linear(1024, 256, initialW=initializer)

            # parames for latent distribution
            if self.is_continuous:
                self.fc2_mu = L.Linear(256, self.latent_cont_dim, initialW=initializer)
                self.fc2_ln_var = L.Linear(256, self.latent_cont_dim, initialW=initializer)
            if self.is_discrete:
                # Linear layer for each categorical distribution
                fc2_alphas = chainer.ChainList()
                for disc_dim in self.latent_spec['disc']:
                    fc2_alphas.add_link(L.Linear(256, disc_dim))
                self.fc2_alphas = fc2_alphas

            # Decoder
            self.fc3 = L.Linear(self.latent_dim, 256, initialW=initializer)
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
        latent_dist = self.encode(x)
        latent_sample = []

        if self.is_continuous:
            mu, ln_var = latent_dist["cont"]
            cont_sample = self.sample_normal(mu, ln_var, mode)
            latent_sample.append(cont_sample)

        if self.is_discrete:
            for alpha in latent_dist["disc"]:
                disc_sample = self.sample_gumbel_softmax(alpha, mode)
                latent_sample.append(disc_sample)

        z = F.concat(latent_sample, axis=1)
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

        latent_dist = {}
        if self.is_continuous:
            latent_dist["cont"] = [self.fc2_mu(h), self.fc2_ln_var(h)]
        if self.is_discrete:
            latent_dist["disc"] = []
            for fc2_alpha in self.fc2_alphas:
                latent_dist["disc"].append(F.softmax(fc2_alpha(h), axis=1))
        return latent_dist

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

    def sample_normal(self, mu, ln_var, mode="mean"):
        """
        Samples from a normal distribution using the reparameterization trick.
        Parameters
        ----------
        mu :
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.
        ln_var :
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        if mode == "mean":
            z = mu
        elif mode == "sample":
            z = F.gaussian(mu, ln_var)
        else:
            raise NotImplementedError
        return z

    def sample_gumbel_softmax(self, alpha, mode="mean", EPS=1e-12):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.
        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D)
        """
        xp = alpha.xp
        batchsize = alpha.shape[0]
        n_dim = alpha.shape[1]
        if mode == "mean":
            # In reconstruction mode, pick most likely sample
            argmax_alpha = F.argmax(alpha, axis=1)
            one_hot_samples = xp.eye(batchsize, n_dim)[argmax_alpha.data]
            return one_hot_samples.astype("float32")
        elif mode == "sample":
            # Sample from gumbel distribution
            unif = xp.random.rand(batchsize, n_dim).astype("float32")
            gumbel = -F.log(-F.log(unif + EPS) + EPS)
            # Reparameterize to create gumbel softmax sample
            log_alpha = F.log(alpha + EPS)
            logit = (log_alpha + gumbel) / self.temperature
            return F.softmax(logit, axis=1)
        else:
            raise NotImplementedError


class JointUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.vae = kwargs.pop('model')
        super(JointUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        vae_optimizer = self.get_optimizer('opt_vae')
        xp = self.vae.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x = chainer.dataset.concat_examples(batch, device=self.device)

        latent_dist = self.vae.encode(x)

        # reconstruction loss
        rec_loss = 0
        for _ in range(self.vae.k):
            reconstructions = self.vae(x, sigmoid=False, mode="sample")
            rec_loss += F.bernoulli_nll(x, reconstructions) \
                / (self.vae.k * batchsize)
        ### latent loss
        # latent loss for continuous
        cont_capacity_loss = 0
        if self.vae.is_continuous:
            mu, ln_var = latent_dist['cont']
            kl_cont_loss = gaussian_kl_divergence(mu, ln_var) / batchsize
            # Anealing loss
            cont_min, cont_max, cont_num_iters, cont_gamma = \
                self.vae.cont_capacity
            cont_cap_now = (cont_max - cont_min) * self.iteration / float(cont_num_iters) + cont_min
            cont_cap_now = min(cont_cap_now, cont_max)
            cont_capacity_loss = cont_gamma * F.absolute(cont_cap_now - kl_cont_loss)

        # latent loss for discrete
        disc_capacity_loss = 0
        if self.vae.is_discrete:
            kl_disc_loss = kl_multiple_discrete_loss(latent_dist['disc'])
            # Anealing loss
            disc_min, disc_max, disc_num_iters, disc_gamma = \
                self.vae.disc_capacity
            disc_cap_now = (disc_max - disc_min) * self.iteration / float(disc_num_iters) + disc_min
            disc_cap_now = min(disc_cap_now, disc_max)
            # Require float conversion here to not end up with numpy float
            disc_theoretical_max = 0
            for disc_dim in self.vae.latent_spec["disc"]:
                disc_theoretical_max += xp.log(disc_dim)
            disc_cap_now = min(disc_cap_now, disc_theoretical_max.astype("float32"))
            disc_capacity_loss = disc_gamma * F.absolute(disc_cap_now - kl_disc_loss)

        joint_vae_loss = rec_loss + cont_capacity_loss + disc_capacity_loss

        self.vae.cleargrads()
        joint_vae_loss.backward()
        vae_optimizer.update()

        chainer.reporter.report({"rec_loss": rec_loss, "cont_loss": cont_capacity_loss,
                                "disc_loss": disc_capacity_loss, "vae_loss": joint_vae_loss, })
        return


def kl_multiple_discrete_loss(alphas):
    """
    Calculates the KL divergence between a set of categorical distributions
    and a set of uniform categorical distributions.
    Parameters
    ----------
    alphas : list
        List of the alpha parameters of a categorical (or gumbel-softmax)
        distribution. For example, if the categorical atent distribution of
        the model has dimensions [2, 5, 10] then alphas will contain 3
        chainer.variable instances with the parameters for each of
        the distributions. Each of these will have shape (N, D).
    Returns
    ----------
    kl_loss:
       sum of KL loss for categorical distributions.
    """
    kl_loss = 0
    for alpha in alphas:
        kl_loss += kl_discrete_loss(alpha)
    return kl_loss


def kl_discrete_loss(alpha, EPS=1e-12):
    """
    Calculates the KL divergence between a categorical distribution and a
    uniform categorical distribution.
    Parameters
    ----------
    alpha : chainer.Variable
        Parameters of the categorical or gumbel-softmax distribution.
        Shape (N, D)
    Returns
    ----------
    kl_loss:
        KL loss for one categorical distribution
    """
    xp = alpha.xp
    disc_dim = alpha.shape[-1]
    # cross entropy cat and uniform
    log_dim = xp.array([xp.log(disc_dim)], dtype="float32")
    # negative entropy for cat
    neg_entropy = F.sum(alpha * F.log(alpha + EPS), axis=1)
    kl_loss = log_dim + F.mean(neg_entropy, axis=0)
    return kl_loss
