# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by GENZITSU at Preferred Networks, Inc.
# to evaluate metrics for chainer trained model.

"""Evaluation protocol to compute metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf
import gin.tf
from tensorflow.python.framework.errors_impl import NotFoundError

from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.evaluation.metrics import beta_vae  # noqa
from disentanglement_lib.evaluation.metrics import dci  # noqa
from disentanglement_lib.evaluation.metrics import downstream_task  # noqa
from disentanglement_lib.evaluation.metrics import factor_vae  # noqa
from disentanglement_lib.evaluation.metrics import irs  # noqa
from disentanglement_lib.evaluation.metrics import mig  # noqa
from disentanglement_lib.evaluation.metrics import modularity_explicitness  # noqa
from disentanglement_lib.evaluation.metrics import reduced_downstream_task  # noqa
from disentanglement_lib.evaluation.metrics import sap_score  # noqa
from disentanglement_lib.evaluation.metrics import unsupervised_metrics  # noqa
from disentanglement_lib.utils import results


def get_dataset_name():
    """Reads the name of the dataset from the environment variable `DATASET_NAME`."""
    return os.getenv("DATASET_NAME", "dsprites_full")


def evaluate_with_gin(model_dir,
                      output_dir,
                      overwrite=False,
                      gin_config_files=None,
                      gin_bindings=None):
    """Evaluate a representation based on the provided gin configuration.

    This function will set the provided gin bindings, call the evaluate()
    function and clear the gin config. Please see the evaluate() for required
    gin bindings.

    Args:
      model_dir: String with path to directory where the representation is saved.
      output_dir: String with the path where the evaluation should be saved.
      overwrite: Boolean indicating whether to overwrite output directory.
      gin_config_files: List of gin config files to load.
      gin_bindings: List of gin bindings to use.
    """
    if gin_config_files is None:
        gin_config_files = []
    if gin_bindings is None:
        gin_bindings = []
    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
    evaluate(model_dir, output_dir, overwrite)
    gin.clear_config()


@gin.configurable(
    "evaluation", blacklist=["model_dir", "output_dir", "overwrite"])
def evaluate(model_dir,
             output_dir,
             overwrite=False,
             evaluation_fn=gin.REQUIRED,
             random_seed=gin.REQUIRED,
             name=""):
    """Loads a representation TFHub module and computes disentanglement metrics.

    Args:
      model_dir: String with path to directory where the representation function
        is saved.
      output_dir: String with the path where the results should be saved.
      overwrite: Boolean indicating whether to overwrite output directory.
      evaluation_fn: Function used to evaluate the representation (see metrics/
        for examples).
      random_seed: Integer with random seed used for training.
      name: Optional string with name of the metric (can be used to name metrics).
    """
    # We do not use the variable 'name'. Instead, it can be used to name scores
    # as it will be part of the saved gin config.
    del name

    # Delete the output directory if it already exists.
    if tf.gfile.IsDirectory(output_dir):
        if overwrite:
            tf.gfile.DeleteRecursively(output_dir)
        else:
            raise ValueError("Directory already exists and overwrite is False.")

    # Set up time to keep track of elapsed time in results.
    experiment_timer = time.time()

    try:
        # Automatically set the proper data set if necessary. We replace the active
        # gin config as this will lead to a valid gin config file where the data set
        # is present.
        if gin.query_parameter("dataset.name") == "auto":
            # Obtain the dataset name from the gin config of the previous step.
            gin_config_file = os.path.join(model_dir, "results", "gin",
                                           "postprocess.gin")
            gin_dict = results.gin_dict(gin_config_file)
            with gin.unlock_config():
                gin.bind_parameter("dataset.name", gin_dict["dataset.name"].replace(
                    "'", ""))
        dataset = named_data.get_named_ground_truth_data()
    except NotFoundError:
        # If we did not train with disentanglement_lib, there is no "previous step",
        # so we'll have to rely on the environment variable.
        if gin.query_parameter("dataset.name") == "auto":
            with gin.unlock_config():
                gin.bind_parameter("dataset.name", get_dataset_name())
        dataset = named_data.get_named_ground_truth_data()

    if os.path.exists(os.path.join(model_dir, 'vae.model')):
        module_path = os.path.join(model_dir, 'vae.model')
        results_dict = _evaluate_with_chainer(module_path, evaluation_fn,
                                              dataset, random_seed)
    else:
        raise RuntimeError("`model_dir` must contain chainer model vae.model .")

    # Save the results (and all previous results in the pipeline) on disk.
    original_results_dir = os.path.join(model_dir, "results")
    results_dir = os.path.join(output_dir, "results")
    results_dict["elapsed_time"] = time.time() - experiment_timer
    results.update_result_directory(results_dir, "evaluation", results_dict,
                                    original_results_dir)


def _evaluate_with_chainer(module_path, evalulation_fn, dataset, random_seed):
    import utils_chainer
    # Load model and make a representor
    model = utils_chainer.import_model(path=module_path)
    _representation_function = utils_chainer.make_representor(model)
    # Evaluate score with the evaluation_fn
    results_dict = evalulation_fn(
        dataset,
        _representation_function,
        random_state=np.random.RandomState(random_seed)
    )
    # Easy peasy lemon squeezy
    return results_dict
