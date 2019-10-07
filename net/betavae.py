from net.shared import BaseVAE


class BetaVAE(BaseVAE):
    """BetaVAE model.
    Based on Equation (4) of "β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A
    CONSTRAINED VARIATIONAL FRAMEWORK"
    (https://openreview.net/references/pdf?id=Sy2fzU9gl)
    """
    def __init__(self, n_z=10, beta=16, k=1, channel_num=3):
        """
        Args:
            n_z (int) :  Latent dimension
            beta (int) : Regularization strength for KL divergence.
            k (int) : Number of Monte Carlo samples used in encoded vector.
        """
        super(BetaVAE, self).__init__(n_z=n_z, beta=beta, k=k,
                                      channel_num=channel_num)
        with self.init_scope():
            self.args = {"n_z": n_z, "beta": beta,
                         "k": k, "channel_num": channel_num}
