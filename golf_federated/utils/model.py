# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/5/22 17:09
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/10/27 1:44

import os
from typing import Tuple
from numpy import ndarray
import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.stats import pearsonr

from golf_federated.utils.log import loggerhear


# def load_model_file(
#         filepath: str,
#         library: str
# ) -> object:
#     """
#
#     Load model from specified file path.
#
#     Args:
#         filepath (str): Path to the file where the model is stored.
#         library (str): The library used to build model.
#
#     Returns:
#         Object: Loaded model.
#
#     """
#
#     # Initialize.
#     file_extension = os.path.splitext(filepath)[-1]
#     loggerhear.log("Common Info  ", 'Loading Model from %s' % filepath)
#
#     # Judge the file format and model library.
#     if file_extension == '.h5' and (library == 'tensorflow' or library == 'keras'):
#         # Load tensorflow or keras model stored in h5 file.
#         from tensorflow.python.keras.models import load_model
#         model = load_model(filepath)
#
#     elif library == 'pytorch' and (file_extension == '.pt' or file_extension == '.pth' or file_extension == '.pkl'):
#         # Load pytorch model stored in pt/pth/pkl file.
#         import torch
#         model = torch.load(filepath)
#
#     else:
#         # When asked to read an unsupported file format or model library, log out and terminate the program.
#         loggerhear.log("Error Message", 'Loading %s Error' % filepath)
#         exit(1)
#
#     return model


# def load_model_parameter(
#         model: object,
#         filepath: str,
#         library: str
# ) -> object:
#     """
#
#     Load model parameters from specified file path.
#
#     Args:
#         model (object): Model before loading parameters.
#         filepath (str): Path to the file where the model parameters is stored.
#         library (str): The library used to build model.
#
#     Returns:
#         Object: Model after loading parameters.
#
#     """
#
#     # Initialize.
#     file_extension = os.path.splitext(filepath)[-1]
#     loggerhear.log("Common Info  ", 'Loading Model Parameter from %s' % filepath)
#
#     # Judge the file format and model library.
#     if file_extension == '.h5' and (library == 'tensorflow' or library == 'keras'):
#         # Load tensorflow or keras model parameters stored in h5 file.
#         model.load_weights(filepath)
#
#     elif library == 'pytorch' and (file_extension == '.pt' or file_extension == '.pth' or file_extension == '.pkl'):
#         # Load pytorch model parameters stored in pt/pth/pkl file.
#         import torch
#         model.load_state_dict(torch.load(filepath))
#
#     else:
#         # When asked to read an unsupported file format or model library, log out and terminate the program.
#         loggerhear.log("Error Message", 'Loading %s Error' % filepath)
#         exit(1)
#
#     return model


def get_model_parameter(
        model: object,
        library: str
) -> list:
    """

    Get parameters of the specified model.

    Args:
        model (object): Model that will get the parameters.
        library (str): The library used to build model.

    Returns:
        List: List of parameters to represent parameters of different layers.

    """

    # Initialize.
    loggerhear.log("Common Info  ", 'Getting Model Parameter')

    # Judge the model library.
    if library == 'tensorflow' or library == 'keras':
        # Get tensorflow or keras model parameters.
        w = model.get_weights()

    elif library == 'pytorch':
        # Get pytorch model parameters.
        w = []
        for key in model.state_dict():
            w.append(model.state_dict()[key].cpu().numpy())

    else:
        # When asked to read an unsupported model library, log out and terminate the program.
        loggerhear.log("Error Message", 'Getting Error')
        exit(1)

    return w


def set_model_parameter(
        model: object,
        w: list,
        library: str
) -> object:
    """

    Set specified parameters to the specified model.

    Args:
        model (object): Model that will be set parameters.
        w (list):  List of parameters to be set to the model.
        library (str): The library used to build model.

    Returns:
        Object: Model after setting parameters.

    """

    # Initialize.
    loggerhear.log("Common Info  ", 'Setting Model Parameter')

    # Judge the model library.
    if library == 'tensorflow' or library == 'keras':
        # Set tensorflow or keras model parameters.
        model.set_weights(w)

    elif library == 'pytorch':
        # Set pytorch model parameters.
        import torch
        idx = 0
        for key in model.state_dict():
            model.state_dict()[key] = torch.from_numpy(w[idx])

    else:
        # When asked to read an unsupported model library, log out and terminate the program.
        loggerhear.log("Error Message", 'Setting Error')
        exit(1)

    return model


# def save_model(
#         model: object,
#         filepath: str,
#         library: str
# ) -> object:
#     """
#
#     Save model parameters to specified file path.
#
#     Args:
#         model (object): Model whose parameters will be saved.
#         filepath (str): Path to the file where the model parameters is to be saved.
#         library (str): The library used to build model.
#
#     Returns:
#         Object: The above model.
#
#     """
#
#     # Initialize.
#     file_name = os.path.splitext(filepath)[0]
#
#     # Judge the file format and model library.
#     if library == 'tensorflow' or library == 'keras':
#         # Save tensorflow or keras model parameters to h5 file.
#         filepath = file_name + '.h5'
#         model.save_weights(filepath)
#
#     elif library == 'pytorch':
#         # Save pytorch model parameters stored in pt/pth/pkl file.
#         import torch
#         filepath = file_name + '.pt'
#         torch.save(model.state_dict(), filepath)
#
#     else:
#         # When asked to read an unsupported file format or model library, log out and terminate the program.
#         loggerhear.log("Error Message", 'Saving to %s Error' % filepath)
#         exit(1)
#
#     loggerhear.log("Common Info  ", 'Saving Model Parameter to %s' % filepath)
#
#     return model


# def compute_rda(
#         model: object,
#         stimulus: ndarray,
#         distance_measure: str,
#         layer_idx: int
# ) -> list:
#     """
#
#     Calculate the representational dissimilarity for specific layer.
#
#     Args:
#         model (object): Model object.
#         stimulus (numpy.ndarray): Stimulus array.
#         distance_measure (str): Field name for distance calculation.
#         layer_idx (int): Layer ID.
#
#     Returns:
#         List: List of representational dissimilarity.
#
#     """
#
#     # Judge the model layer name and perform different operations.
#     if 'conv2d' in model.layers[layer_idx].name or 'dense' in model.layers[layer_idx].name:
#         from tensorflow.python.keras.models import Model
#         layer_model = Model(inputs=model.input, outputs=model.layers[layer_idx].output)
#         feature_all = []
#         len_sti = len(stimulus)
#         for j in range(len_sti):
#             x_stimulu = np.expand_dims(stimulus[j], axis=0)
#             feature = layer_model.predict(x_stimulu)
#             feature = np.array(feature).reshape(-1, feature.size)
#             feature_all.append(feature)
#         feature_all = np.array(feature_all)
#         rdms_one_layer = squareform(pdist(feature_all.reshape(len_sti, -1), distance_measure))
#         result = [np.nan_to_num(rdms_one_layer[0])]
#         if '/Softmax' in model.layers[layer_idx].output.name or '/Relu' in model.layers[layer_idx].output.name:
#             result.append(0)
#
#     elif 'batch_normalization' in model.layers[layer_idx].name:
#         result = [0 for _ in range(4)]
#
#     elif 'max_pooling' in model.layers[layer_idx].name or 'dropout' in model.layers[layer_idx].name or 'flatten' in \
#             model.layers[layer_idx].name:
#         result = []
#
#     else:
#         result = [0]
#
#     return result


# def compute_rda_alllayers(
#         layer_num: int,
#         model: object,
#         data_stimulus: ndarray,
#         distance_measure: str,
# ) -> list:
#     """
#
#     Calculate the representational dissimilarity for all layers.
#
#     Args:
#         layer_num (int): Number of layers.
#         model (object): Model object.
#         data_stimulus (numpy.ndarray): Stimulus array.
#         distance_measure (str): Field name for distance calculation.
#
#     Returns:
#         List: List of representational dissimilarity.
#
#     """
#
#     global_rdm_alllayer = []
#     layer_idx_all = layer_num
#
#     for layer_idx in range(layer_idx_all):
#         global_rdm_onelayer = compute_rda(model, data_stimulus, distance_measure,
#                                           layer_idx)
#         global_rdm_alllayer += global_rdm_onelayer
#
#     return global_rdm_alllayer


# def compute_rc_simp(
#         rda1: ndarray,
#         rda2: ndarray
# ) -> Tuple[float, float]:
#     """
#
#     Calculate the pearson correlation coefficient.
#
#     Args:
#         rda1 (numpy.ndarray): First array for calculation.
#         rda2 (numpy.ndarray): Second array for calculation.
#
#     Returns:
#         Tuple[float, float]): Two-dimensional data, respectively, Pearson product-moment correlation coefficient and P-value.
#
#     """
#
#     pccs = pearsonr(rda1, rda2)
#
#     return pccs
