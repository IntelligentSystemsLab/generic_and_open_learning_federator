# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 12:43
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 12:43
from copy import deepcopy
from typing import Callable, List
import numpy as np
import math
import pandas as pd

from golf_federated.utils.data import deepcopy_list


def fedavg(
    weight: List,
    data_size: List
) -> List:
    """

    Function implementation of FedAVG, which directly averages the corresponding values of collected model parameters.

    Args:
        weight (List): List of models to aggregate.
        data_size (List): List of data sizes of clients.

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
    client_id: List,
    weight: dict,
    client_round: dict,
    version_latest: int,
) -> List:
    """

    Function implementation of FedFD, which weighted averages the corresponding values of collected model parameters.

    Args:
        client_id (List): ID of clients that upload the models.
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
    client_id: List,
    weight: dict,
    staleness: str,
    current_weight: List,
    current_round: int,
    client_round: dict,
    alpha: float,
    beta: float
) -> List:
    """

    Function implementation of FedAsync.

    Args:
        client_id (List): List of uploaded client names.
        weight (dict): Dict of uploaded local model weight.
        staleness (str): Corresponds to the name of the function defined in FedAsync.
        current_weight (List): Current global model parameters.
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
    weight: List,
    aggregate_percentage: List,
    current_weight: List
) -> List:
    """

    Function implementation of SLMFed_Sync.

    Args:
        weight (List): List of client model parameters for aggregation.
        aggregate_percentage (List): Aggregate weights for each client.
        current_weight (List): Current global model parameters.

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
    client_id: List,
    weight: dict,
    aggregate_percentage: dict,
    current_weight: List,
    current_acc: float,
    target_acc: float,
    func: str
) -> List:
    """

    Function implementation of SLMFed_Async.

    Args:
        client_id (List): List of client IDs for aggregation.
        weight (dict): Dictionary of client model parameters for aggregation.
        aggregate_percentage (dict): Aggregate weights for each client.
        current_weight (List): Current global model parameters.
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
    w_global: List,
    w_local: List,
    miu: float = 1
) -> Callable:
    """

    The optimized loss function defined in FedProx.

    Args:
        model_library (str): The library used to build model.
        w_global (List): Global model.
        w_local (List): Local model.
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


def mean2(x):
    import torch
    y = torch.sum(x) / len(x)
    return y


def corr2(a, b):
    a = a - mean2(a)
    b = b - mean2(b)
    import torch
    r = torch.sum(a * b) / torch.sqrt(torch.sum(a * a) * torch.sum(b * b))
    return r


import torch
from torch.autograd import Variable


def Cedarsyn(
    local_model: List,
    detect: bool,
    current_model: object,
    num_layer: int,
    layer_weight: List,
    stimulus_x: torch.tensor,
    require_judge_layer: List,
    upgrade_bool_list: pd.DataFrame,
) -> object:
    """

    Function implementation of Cedar_Sync.

    Args:
        local_model (List): List of client model parameters for aggregation.
        detect (bool): Whether to detect malicious updates.
        current_model (object): Current global model.
        num_layer (int): Number of layers involved in filtering.
        layer_weight (List): Weights to measure the importance of layers.
        stimulus_x (torch.tensor): Stimuli.
        require_judge_layer (List): Layers involved in filtering.
        upgrade_bool_list (pandas.DataFrame): Matrix of whether the layer is uploaded.

    Returns:
        object: Updated model.

    """

    require_fill_layer = []
    for i in range(num_layer):
        if layer_weight[i] == 'inf':
            require_fill_layer.append(require_judge_layer[i])

    global_model_copy = deepcopy(current_model)

    aggregate_percentage = []

    RS = np.zeros([1, len(local_model)])
    from torch.nn import functional as F
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i_local_model in range(len(local_model)):
        stimulus_num = 10
        temp_model1 = deepcopy(local_model[i_local_model])
        temp_model2 = deepcopy(current_model)
        model1_out = temp_model1(stimulus_x)
        model2_out = temp_model2(stimulus_x)
        model1_softmax = torch.tensor(F.softmax(model1_out, dim=1), dtype=torch.float32)
        model2_softmax = torch.tensor(F.softmax(model2_out, dim=1), dtype=torch.float32)
        model1_RDM = torch.zeros((stimulus_num, stimulus_num)).to(device)
        model2_RDM = torch.zeros((stimulus_num, stimulus_num)).to(device)

        for i in range(stimulus_num):
            for j in range(stimulus_num):
                temp_a = model1_softmax[i]
                temp_c = temp_a.view(1, -1)
                temp_b = model1_softmax[j]
                temp_d = temp_b.view(1, -1)
                model1_RDM[i][j] = torch.cosine_similarity(temp_c, temp_d)
                model2_RDM[i][j] = torch.cosine_similarity(model2_softmax[i].view(1, -1), model2_softmax[j].view(1, -1))
        corr_result = corr2(model1_RDM, model2_RDM)
        RS[0][i_local_model] = corr_result
        aggregate_percentage.append(float(corr_result))
    if detect:
        RS = (RS - RS.min()) / (RS.max() - RS.min())
        from sklearn.cluster import KMeans
        cluster = KMeans(n_clusters=2)
        cluster_result = cluster.fit(RS.reshape(-1, 1)).labels_
        from collections import Counter
        cluster_benign = Counter(cluster_result).most_common(1)[0][0]
        id_train = []
        for i in range(len(local_model)):
            if cluster_result[i] == cluster_benign:
                id_train.append(i)
                aggregate_percentage[i] = 0
    sum_aggregate_percentage = sum(aggregate_percentage)
    for a in range(len(aggregate_percentage)):
        aggregate_percentage[a] = aggregate_percentage[a] / sum_aggregate_percentage
    aggregate_percentage = np.array(aggregate_percentage)

    for j in range(len(local_model)):
        for global_w, client_w in zip(current_model.named_parameters(), local_model[j].named_parameters()):
            global_name, global_param = global_w
            client_name, client_param = client_w
            for layer in require_judge_layer:
                if global_name.__contains__(layer):
                    judge_sign = 1
                    break
                else:
                    judge_sign = 0
            if judge_sign == 0:
                if j == 0:
                    param_tem = Variable(torch.zeros_like(global_param))
                    global_param.data.copy_(param_tem.data)
                global_param.data.add_(client_param.data * aggregate_percentage[j])
            else:
                if j == 0:
                    param_tem = Variable(torch.zeros_like(global_param))
                    global_param.data.copy_(param_tem.data)
                for layer in require_judge_layer:
                    if global_name.__contains__(layer):
                        idx = list(require_judge_layer).index(layer)
                if upgrade_bool_list[j][idx] == 1:
                    global_param.data.add_(client_param.data * layer_weight[idx])
    for layer in require_fill_layer:
        for global_w, global_copy in zip(current_model.named_parameters(), global_model_copy.named_parameters()):
            global_name, global_param = global_w
            global_copy_name, global_copy_param = global_copy
            if global_name.__contains__(layer):
                global_param.data.add_(global_copy_param.data * 1)

    return current_model
