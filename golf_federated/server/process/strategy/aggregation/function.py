# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 12:43
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 12:43

from typing import Callable
import numpy as np
import math

from golf_federated.utils.data import deepcopy_list


def fedavg(
        weight: list,
        data_size: list
) -> list:
    """

    Function implementation of FedAVG, which directly averages the corresponding values of collected model parameters.

    Args:
        weight (list): List of models to aggregate.
        data_size (list): List of data sizes of clients.

    Returns:
        List: The model generated after aggregation. And use a list to store the parameters of different layers.

    """

    # Number of clients uploading local model parameters.
    num = len(weight)

    # Use a loop to calculate the sum of each model parameter value, and then take the average.
    result = np.array(weight[0]) * data_size[0]
    for i in range(1, num):
        w = np.array(weight[i])
        d = data_size[i]
        result += w * d
    return_result = (result / sum(data_size)).tolist()

    return return_result


def fedfd(
        client_id: list,
        weight: dict,
        client_round: dict,
        version_latest: int,
) -> list:
    """

    Function implementation of FedFD, which weighted averages the corresponding values of collected model parameters.

    Args:
        client_id (list): ID of clients that upload the models.
        weight (dict): Corresponding dictionary of client IDs and models to aggregate.
        client_round (dict): Corresponding dictionary of client IDs and number of training rounds for local models.
        version_latest (int): Latest model version.

    Returns:
        List: The model generated after aggregation. And use a list to store the parameters of different layers.

    """

    # Use a loop to calculate the weighted sum of each model parameter value, and then take the average.
    total = 0
    w = 0
    for c in client_id:
        total += (version_latest - client_round[c] + 1) ** (-0.5)
        try:
            weight_c = weight[c]
        except:
            continue
        theta = ((version_latest - client_round[c] + 1) ** (-0.5))
        w += theta * np.array(weight_c)
    global_model = w / total
    return_result = global_model.tolist()

    return return_result


def fedasync(
        client_id: list,
        weight: dict,
        staleness: str,
        current_weight: list,
        current_round: int,
        client_round: dict,
        alpha: float,
        beta: float
) -> list:
    """

    Args:
        client_id (list): List of uploaded client names.
        weight (dict): Dict of uploaded local model weight.
        staleness (str): Corresponds to the name of the function defined in FedAsync.
        current_weight (list): Current global model parameters.
        current_round (int): Number of current training round.
        client_round (dict): Number of global round corresponding to the model trained by each client.
        alpha (float): Corresponds to the parameter α defined in FedAsync.
        beta (float): Corresponds to the parameter β defined in FedAsync.

    Returns:
        List: The model generated after aggregation. And use a list to store the parameters of different layers.

    """

    # Initialize temporary variables.
    alpha_clients = []
    weight_list = deepcopy_list(current_weight)
    return_result = deepcopy_list(current_weight)
    first = True
    layer_num = len(current_weight)

    # For all uploaded clients, model parameters are weightily summarized.
    for c_id in client_id:
        c_weight = deepcopy_list(weight[c_id])
        c_round = client_round[c_id]
        if staleness == 'Linear':
            s = 1 / (alpha * (current_round - c_round) + 1)
        elif staleness == 'Polynomial':
            s = math.pow(current_round - c_round + 1, (-alpha))
        elif staleness == 'Exponential':
            s = math.exp(-alpha * (current_round - c_round))
        elif staleness == 'Hinge':
            if current_round - c_round <= beta:
                s = 1
            else:
                s = 1 / (alpha * (current_round - c_round - beta))
        else:
            s = 1
        alpha_c = s * alpha
        alpha_clients.append(alpha_c)
        for l in range(layer_num):
            if first:
                weight_list[l] = c_weight[l] * alpha_c

            else:
                weight_list[l] += c_weight[l] * alpha_c
        first = False

    # The summarized model parameters are averaged to obtain the global model parameters.
    avg_alpha = sum(alpha_clients) / len(alpha_clients)
    for l in range(len(current_weight)):
        return_result[l] = (1 - avg_alpha) * current_weight[l] + avg_alpha * weight_list[l] / sum(alpha_clients)

    return return_result


def SLMFedsyn(
        weight: list,
        aggregate_percentage: list,
        current_weight: list
) -> list:
    """

    Args:
        weight (list): List of client model parameters for aggregation.
        aggregate_percentage (list): Aggregate weights for each client.
        current_weight (list): Current global model parameters.

    Returns:
        List: The model generated after aggregation. And use a list to store the parameters of different layers.

    """

    # Initialize temporary variables.
    aggregate_percentage_array = np.array(aggregate_percentage)
    weight_array = np.array(weight)
    client_num = len(weight)
    layer_num = len(weight[0])
    first = [0 for i in range(layer_num)]
    return_result = deepcopy_list(current_weight)

    # Update the parameters for each layer of the model separately.
    for l in range(layer_num):
        none_client = []

        # Calculate parameters for all clients.
        for i in range(client_num):
            if weight_array[i][l] is None:
                none_client.append(i)
                continue
            else:
                content = weight_array[i][l] * aggregate_percentage_array[i]
                if first[l] == 0:
                    return_result[l] = content
                    first[l] = 1
                else:
                    return_result[l] += content

        # Adjust the parameters according to the proportion of None
        if 0 < len(none_client) < client_num:
            none_prob = 0
            for i in none_client:
                none_prob += aggregate_percentage_array[i]
            return_result[l] = return_result[l] / (1 - none_prob)

    return return_result


def SLMFedasyn(
        client_id: list,
        weight: dict,
        aggregate_percentage: dict,
        current_weight: list,
        current_acc: float,
        target_acc: float,
        func: str
) -> list:
    """

    Args:
        client_id (list): List of client IDs for aggregation.
        weight (dict): Dictionary of client model parameters for aggregation.
        aggregate_percentage (dict): Aggregate weights for each client.
        current_weight (list): Current global model parameters.
        current_acc (float): Current accuracy corresponding to the global model.
        target_acc (float): Target accuracy of the task.
        func (str):  Function to adjust aggregation weights. Default as 'other'.

    Returns:
        List: The model generated after aggregation. And use a list to store the parameters of different layers.

    """

    # Initialize temporary variables.
    layer_num = len(weight[client_id[0]])
    return_result = deepcopy_list(current_weight)
    upload_content = deepcopy_list(current_weight)
    first = [0 for i in range(layer_num)]
    if func == 'linear':
        alpha = current_acc / target_acc
    elif func == 'concave_exp':
        alpha = 1 - math.exp(-2 * math.e * current_acc / target_acc)
    elif func == 'convex_quadratic':
        alpha = (current_acc / target_acc) ** 2
    elif func == 'concave_quadratic':
        alpha = 1 - (current_acc / target_acc - 1) ** 2
    elif func == 'convex_exp':
        alpha = (math.exp(current_acc / target_acc) - 1) / (math.e - 1)
    else:
        alpha = 0.5

    # Update the parameters for each layer of the model separately.
    for l in range(layer_num):
        p_sum = 0

        # Calculate parameters for all clients.
        for id in client_id:
            if weight[id][l] is None:
                continue
            else:
                p = aggregate_percentage[id]
                p_sum += p
                content = weight[id][l] * p
                if first[l] == 0:
                    upload_content[l] = content
                    first[l] = 1
                else:
                    upload_content[l] += content

        # Adjust parameters according to established rules.
        if p_sum != 0:
            q_new = (1 - p_sum) * alpha
            new_content = upload_content[l] / p_sum
            old_content = return_result[l]
            return_result[l] = new_content * (1 - q_new) + old_content * q_new

    return return_result


def fedprox_loss(
        model_library: str,
        w_global: list,
        w_local: list,
        miu: float = 1
) -> Callable:
    """

    The optimized loss function defined in FedProx.

    Args:
        model_library (str): The library used to build model.
        w_global (list): Global model.
        w_local (list): Local model.
        miu (float): Corresponds to the parameter μ defined in FedProx. Default as 1.

    Returns:
        Callable: Loss function.

    """

    def loss(y_true, y_pred):

        proximal_term = 0
        for w_g, w_l in zip(w_global, w_local):
            proximal_term += np.linalg.norm([w_g, w_l])
        if model_library == 'tensorflow' or model_library == 'keras':
            from tensorflow.keras.losses import sparse_categorical_crossentropy
            return sparse_categorical_crossentropy(y_true, y_pred) + miu / 2 * proximal_term

    return loss
