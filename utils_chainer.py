import os
from copy import deepcopy
from collections import namedtuple
from matplotlib import pyplot as plt

import numpy as np
from PIL import Image
import chainer
from chainer import serializers
from chainer.dataset import DatasetMixin
from chainer.iterators import SerialIterator

from net.betavae import BetaVAE
from net.factorvae import FactorVAE
from net.dipvae import DIPVAE
from net.jointvae import JointVAE
from net.shared import BaseVAE

# ------ Data Loading ------
if 'DISENTANGLEMENT_LIB_DATA' not in os.environ:
    os.environ.update({'DISENTANGLEMENT_LIB_DATA': os.path.join(os.path.dirname(__file__),
                                                                'dataset')})
# noinspection PyUnresolvedReferences
from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data
# --------------------------


ExperimentConfig = namedtuple('ExperimentConfig',
                              ('base_path', 'experiment_name', 'dataset_name'))


def get_config():
    """
    This function reads the environment variables OUTPUT_PATH,
    EVALUATION_NAME and DATASET_NAME and returns a named tuple.
    """
    return ExperimentConfig(base_path=os.getenv("OUTPUT_PATH", "./results"),
                            experiment_name=os.getenv("EVALUATION_NAME", "dev_tmp"),
                            dataset_name=os.getenv("DATASET_NAME", "dsprites_full"))


def get_dataset_name():
    """Reads the name of the dataset from the environment variable `DATASET_NAME`."""
    return os.getenv("DATASET_NAME", "dsprites_full")


def get_model_path(base_path=None, experiment_name=None, make=True):
    """
    This function gets the path to where the model is expected to be stored.

    Parameters
    ----------
    base_path : str
        Path to the directory where the experiments are to be stored.
        This defaults to OUTPUT_PATH (see `get_config` above) and which in turn
        defaults to './results'.
    experiment_name : str
        Name of the experiment. This defaults to EVALUATION_NAME which in turn
        defaults to 'experiment_name'.
    make : Makes the directory where the returned path leads to (if it doesn't exist already)

    Returns
    -------
    str
        Path to where the model should be stored (to be found by the evaluation function later).
    """
    base_path = os.getenv("OUTPUT_PATH", "./results") \
        if base_path is None else base_path
    experiment_name = os.getenv("EVALUATION_NAME", "dev_tmp") \
        if experiment_name is None else experiment_name
    model_path = os.path.join(base_path, experiment_name, 'representation', 'vae.model')
    if make:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(model_path), 'results'), exist_ok=True)
    return model_path


def export_model(model, path=None):
    """
    Parameters
    ----------
    model : chainer.Chain
        trained chainer model
    path : str
        Path to the file where the model is saved. Defaults to the value set by the
        `get_model_path` function above.
    Returns
    -------
    str
        Path to where the model is saved.
    Note
    -------
    Must import model from net in advance
    """
    model_path = get_model_path() if path is None else path
    model_name = model.__class__.__name__
    # export model config
    config_path = "/".join(model_path.split("/")[:-1])
    with open(config_path + "/model.config", mode='w') as f:
        f.write(f"{model_name}(**{model.args})")
    model = deepcopy(model).to_cpu()
    serializers.save_npz(model_path, model)
    return path


def import_model(path=None):
    """
    Imports a model as chainer.Chain from file.
    You must import adequate model in advance.
    Parameters
    ----------
    path : str
        Path to where the model is saved.
        Defaults to the return value of the `get_model_path`
    Returns
    -------
    chainer.Chain
    """
    model_path = get_model_path() if path is None else path

    # load model config
    config_path = "/".join(model_path.split("/")[:-1])
    with open(config_path + "/model.config", mode='r') as f:
        model_config = f.read()

    # load model
    model = eval(model_config)
    serializers.load_npz(get_model_path(), model)
    return model


def make_representor(model, flat=False):
    """
    Encloses the chainer model in a callable that can be used by `disentanglement_lib`.
    this function need for local evaluation
    Parameters
    ----------
    model : chainer.Chain
        The chainer model.
    Returns
    -------
    callable
        A callable function (`representation_function` in dlib code)
    """
    model = model.to_cpu()

    # Define the representation function
    def _represent(x):
        assert isinstance(x, np.ndarray), \
            "Input to the representation function must be a ndarray."
        x = x.astype("float32")
        assert x.ndim == 4, \
            "Input to the representation function must be a four dimensional NHWC tensor."
        # Convert from NHWC to NCHW
        x = np.moveaxis(x, 3, 1)
        if flat:
            x = x.reshape(-1, 3 * 64 * 64)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            y = model(x, represent=True, sigmoid=False)
        y = y.data
        assert y.ndim == 2, \
            "The returned output from the representor must be two dimensional (NC)."
        return y

    return _represent


def get_loader(name=None, batch_size=256, flat=False):
    """
    Makes a dataset and a data-loader.
    Parameters
    ----------
    name : str
        Name of the dataset use. Defaults to the output of `get_dataset_name`.
    batch_size : int
        Batch size.
    Returns
    -------
    SerialIterator
    """
    name = get_dataset_name() if name is None else name
    dataset = get_named_ground_truth_data(name)
    assert isinstance(dataset.images, np.ndarray), \
        "Input to the representation function must be a ndarray."
    if dataset.images.ndim == 3:
        # expand channel dim
        dataset.images = np.expand_dims(dataset.images, axis=3)
    assert dataset.images.ndim == 4, \
        "Input to the representation function must be a four dimensional NHWC tensor."
    # Convert from NHWC to NCHW
    dataset.images = np.moveaxis(dataset.images, 3, 1)
    if flat:
        dataset.images = dataset.images.reshape(-1, 3 * 64 * 64)
    cast_dataset = CastDataset(dataset.images)
    loader = SerialIterator(cast_dataset, batch_size=batch_size,
                            repeat=True, shuffle=True)
    return loader


def get_correlated_loader(name=None, batch_size=256, flat=False):
    """
    Make a correlated dataset and a data-loader.
    Parameters
    ----------
    name : str
        Name of the dataset use. Defaults to the output of `get_dataset_name`.
    batch_size : int
        Batch size.
    Returns
    -------
    SerialIterator
    """
    name = get_dataset_name() if name is None else name
    dataset = get_named_ground_truth_data(name)
    # make correlation between x_pos and size
    dataset.images = dataset.images.reshape(3, 6, 40, 32, 32, 64, 64)
    partial_images = []
    for shape in range(3):
        for size in range(6):
            for rotation in range(40):
                for x_pos in range(32):
                    for y_pos in range(32):
                        if size <= 1:
                            if x_pos <= 15:
                                partial_images.append(dataset.images[shape, size, rotation, x_pos, y_pos])
                        if 1 < size and size <= 3:
                            if 7 < x_pos <= 23:
                                partial_images.append(dataset.images[shape, size, rotation, x_pos, y_pos])
                        if 3 < size:
                            if 15 < x_pos:
                                partial_images.append(dataset.images[shape, size, rotation, x_pos, y_pos])
    dataset.images = partial_images
    dataset.images = np.array(dataset.images)

    assert isinstance(dataset.images, np.ndarray), \
        "Input to the representation function must be a ndarray."
    if dataset.images.ndim == 3:
        # expand channel dim
        dataset.images = np.expand_dims(dataset.images, axis=3)
    assert dataset.images.ndim == 4, \
        "Input to the representation function must be a four dimensional NHWC tensor."
    # Convert from NHWC to NCHW
    dataset.images = np.moveaxis(dataset.images, 3, 1)
    if flat:
        dataset.images = dataset.images.reshape(-1, 3 * 64 * 64)

    cast_dataset = CastDataset(dataset.images)
    loader = SerialIterator(cast_dataset, batch_size=batch_size,
                            repeat=True, shuffle=True)
    return loader


def get_missing_loader(name=None, batch_size=256, flat=False):
    """
    Make a missing dataset and a data-loader.
    Parameters
    ----------
    name : str
        Name of the dataset use. Defaults to the output of `get_dataset_name`.
    batch_size : int
        Batch size.
    Returns
    -------
    SerialIterator
    """
    name = get_dataset_name() if name is None else name
    dataset = get_named_ground_truth_data(name)
    # erase random factor pairs
    dataset.images = dataset.images.reshape(3, 6, 40, 32, 32, 64, 64)
    partial_images = []
    for shape in range(3):
        for size in range(6):
            for rotation in range(40):
                for x_pos in range(32):
                    for y_pos in range(32):
                        if np.random.rand() < 0.5:
                            partial_images.append(dataset.images[shape, size, rotation, x_pos, y_pos])
    dataset.images = partial_images
    dataset.images = np.array(dataset.images)

    assert isinstance(dataset.images, np.ndarray), \
        "Input to the representation function must be a ndarray."
    if dataset.images.ndim == 3:
        # expand channel dim
        dataset.images = np.expand_dims(dataset.images, axis=3)
    assert dataset.images.ndim == 4, \
        "Input to the representation function must be a four dimensional NHWC tensor."
    # Convert from NHWC to NCHW
    dataset.images = np.moveaxis(dataset.images, 3, 1)
    if flat:
        dataset.images = dataset.images.reshape(-1, 3 * 64 * 64)

    cast_dataset = CastDataset(dataset.images)
    loader = SerialIterator(cast_dataset, batch_size=batch_size,
                            repeat=True, shuffle=True)
    return loader


def compare_rec_images(model, x, filename):
    """
    Make reconstruction comparison image and
    save it at the experiment directory
    Parameters
    ----------
    model : chainer.Chain
        model for making latent traversal
    x : np.array [1, 3, 64, 64] or [1, 1, 64, 64]
        source image for latent traversal
    filename : str
        file name
    Returns
    -------
    None
    """
    base_path = os.getenv("OUTPUT_PATH", "../results")
    experiment_name = os.getenv("EVALUATION_NAME", "dev_tmp")
    img_path = os.path.join(base_path, experiment_name, filename)
    num = x.data.shape[0]
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        x_ = model(x)
    x = np.r_[x.data, x_.data]
    fig, ax = plt.subplots(2, num, figsize=(num, 2), dpi=100)
    for ai, xi in zip(ax.flatten(), x):
        assert xi.shape[0] == 1 or xi.shape[0] == 3, \
            "channel num should be 1 or 3"
        if xi.shape[0] == 1:
            xi = xi.reshape(64, 64)
        elif xi.shape[0] == 3:
            xi = np.moveaxis(xi, 0, 2)
        ai.tick_params(labelbottom=False, bottom=False)  # remove x-axis ticks
        ai.tick_params(labelleft=False, left=False)  # remove y-axis ticks
        ai.imshow(xi)
    fig.savefig(img_path, bbox_inches='tight', pad_inches=0)
    return


def show_latent_gif(model, x, filename="latent_traversal", show_num=20, z_min=-1.5, z_max=1.5):
    """
    Make latent traversal gif image and save it at the experiment directory
    this function doesn't not work for JointVAE
    Parameters
    ----------
    model : chainer.Chain
        model for making latent traversal
    x : np.array [1, 3, 64, 64] or [1, 1, 64, 64]
        source image for latent traversal
    filename : str
        file name
    show_num : int
        number of images for making gif
    z_min : float
        define latent dim's moving range
    z_max : float
        define latent dim's moving range
    Returns
    -------
    None
    """
    base_path = os.getenv("OUTPUT_PATH", "../results")
    experiment_name = os.getenv("EVALUATION_NAME", "dev_tmp")
    dir_path = os.path.join(base_path, experiment_name, "latent_traversals")
    os.makedirs(dir_path, exist_ok=True)

    # encode
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        latents = model(x, represent=True)

    latent_dim = latents.shape[1]
    z = latents.data[0]
    gif = []
    for z_ij in np.linspace(z_min, z_max, show_num):
        latent_traversals = []
        latent_traversals.append(z)
        for i in range(latent_dim):
            z_ = np.array(z)
            z_[i] = z_ij
            latent_traversals.append(z_)
        latent_traversals = np.array(latent_traversals)
        z_ = chainer.Variable(latent_traversals.astype("float32"))
        x_ = model.decode(z_)
        gif.append(np.concatenate([x.data, x_.data]))

    for i, imgs in enumerate(gif):
        # plot image
        fig, ax = plt.subplots(1, latent_dim + 2,
                               figsize=(latent_dim + 2, 1), dpi=100)
        for ai, xi in zip(ax.flatten(), imgs):
            assert xi.shape[0] == 1 or xi.shape[0] == 3, \
                "channel num should be 1 or 3"
            if xi.shape[0] == 1:
                xi = xi.reshape(64, 64)
            elif xi.shape[0] == 3:
                xi = np.moveaxis(xi, 0, 2)
            ai.tick_params(labelbottom=False, bottom=False)  # remove x-axis ticks
            ai.tick_params(labelleft=False, left=False)  # remove y-axis ticks
            ai.imshow(xi)
        fig.savefig(dir_path + "/" + str(i), bbox_inches='tight', pad_inches=0)
        plt.close()

    imgs = []
    for i in range(show_num):
        imgs.append(Image.open(dir_path + "/" + str(i) + ".png"))

    imgs[0].save(os.path.join(base_path, experiment_name) + "/" + filename + ".gif",
                 save_all=True, append_images=imgs[1:], optimize=False, duration=50, loop=20)
    return


class CastDataset(DatasetMixin):
    """
    Make dataset for Serial Iterator.
    This dataset cast the image type at sampling for memory saving
    """
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def get_example(self, i):
        if self.values[i].max() > 1.:
            return (self.values[i] / 255).astype(np.float32)
        else:
            return self.values[i].astype(np.float32)


if __name__ == '__main__':
    pass
