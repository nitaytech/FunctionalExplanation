import numpy as np
import torch
from torch.nn import Module
from torch import Tensor
from transformers import PreTrainedModel
from typing import List, Tuple, Union, Any, Dict


def to_n_diagonal_matrix(matrix: Tensor, n: int):
    """
    Converts a torch matrix to a n-diagonal matrix.
    """
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    diagonal = torch.diag(torch.diag(matrix, 0), 0)
    for i in range(n):
        diagonal += torch.diag(torch.diag(matrix, i), i) + torch.diag(torch.diag(matrix, -i), i)
    return diagonal


def broadcast_trailing_dims(to_broadcast: Tensor, like_this: Tensor):
    new_shape = [1 for _ in like_this.shape]
    new_shape[:to_broadcast.ndim] = list(to_broadcast.shape)
    return to_broadcast.reshape(*new_shape)


def broadcast_inner_dims(to_broadcast: Tensor, like_this: Tensor):
    new_shape = [1 for _ in like_this.shape]
    old_shape = list(to_broadcast.shape)
    new_shape[-1] = old_shape[-1]
    new_shape[:len(old_shape)-1] = old_shape[:-1]
    return to_broadcast.reshape(*new_shape)


def calculate_covariance(inputs: Tensor):
    x = inputs - inputs.mean(dim=0)
    return torch.mm(x.T, x) / (inputs.shape[0] - 1)


def flatten(x: Tensor, ndims_keep: int = 0):
    if ndims_keep > 0:
        return x.reshape(*(list(x.shape)[:ndims_keep] + [-1]))
    else:
        return x.reshape(-1)


def cholesky(cov: Tensor):
    try:
        return torch.linalg.cholesky(cov)
    except:
        d = torch.abs(torch.diag(cov))
        try:
            return torch.linalg.cholesky(cov + torch.eye(cov.shape[0]) * d.min())
        except:
            return torch.linalg.cholesky(cov + torch.eye(cov.shape[0]) * d.mean())


def _handle_covariance(variance: Union[int, float, np.ndarray, Tensor] = None,
                       inputs: Union[Tensor, np.ndarray] = None,
                       per_channel: bool = False,
                       var_spread: float = 0.15) -> Tensor:
    # can be either None (then estimate the variance over each channel/token), a scalar,
    # a vector (of the length of the number of channels), a covariance matrix
    # or a tensor (in that case, the first dim is the channel/token dim)
    if variance is None and inputs is None:
        variance = torch.tensor(1.0)
    elif variance is None:
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs)
        dims = list(range(1 if per_channel else 0, inputs.ndim))
        variance = torch.var(inputs, dim=dims)
    else:
        if isinstance(variance, (int, float, np.ndarray)):
            variance = torch.tensor(variance)
    if var_spread is not None:
        variance = variance * var_spread
    return variance


def _standard_perturbate(inputs: Tensor,
                         variance: Tensor = None,
                         device: str = None):
    # mean.shape = scalar or (flatten_features,)
    # variance.shape = scalar
    if device is None:
        device = inputs.device
    simulation = (variance ** 0.5) * torch.randn(*inputs.shape, device=device, dtype=inputs.dtype)
    return inputs + simulation


def perturbate(inputs: Tensor,
               covariance: Union[int, float, np.ndarray, Tensor] = None,
               cholesky_decomposed_covariance: Union[np.ndarray, Tensor] = None,
               per_channel: bool = False,
               var_spread: float = 0.15,
               device: str = None):
    inputs_shape = inputs.shape
    if device is None:
        device = inputs.device
    if inputs.ndim > 2:
        inputs = flatten(inputs, ndims_keep=1)
    cdc = cholesky_decomposed_covariance
    covariance = _handle_covariance(covariance, inputs, per_channel, var_spread)
    if var_spread is not None and cdc is not None:
        cdc *= var_spread ** 0.5
    is_variance = covariance is None or isinstance(covariance, (int, float)) or covariance.ndim < 2
    if cdc is None and is_variance:
        return _standard_perturbate(inputs, covariance, device).reshape(*inputs_shape)
    elif cdc is None:
        if isinstance(covariance, np.ndarray):
            covariance = torch.tensor(covariance)
        # a covariance matrix
        if covariance.ndim == 2:
            cdc = cholesky(covariance)
        # a tensor, first dim is channels/tokens
        else:
            cdc = torch.stack([cholesky(cov_i) for cov_i in covariance], dim=0)
    simulation = torch.randn(*inputs.shape, device=device, dtype=inputs.dtype)
    if cdc.ndim == 2:
        simulation = torch.einsum(f"ij, cj -> ci", cdc, simulation)
    else:
        simulation = torch.einsum(f"cij, cj -> ci", cdc, simulation)
    simulation = inputs + simulation
    return simulation.reshape(*inputs_shape)


def extract_gradients(model: Module,
                      inputs: Tensor,
                      modality: str = 'image',
                      attention_mask: Tensor = None,
                      label: Union[int, List[int]] = None,
                      outputs_activation_func=None,
                      return_outputs: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    device = next(model.parameters()).device
    # add batch dim to inputs
    inputs = inputs.unsqueeze(0)
    if modality == 'text':
        assert inputs.ndim == 3
        inputs = inputs.clone().detach().requires_grad_(True).to(device)
        if isinstance(model, PreTrainedModel):
            attention_mask = attention_mask.clone().detach().requires_grad_(False).to(device)
            outputs = model(inputs_embeds=inputs, attention_mask=attention_mask)
            outputs = outputs[0]
        else:
            outputs = model(inputs_embeds=inputs)
    else:
        if modality == 'image':
            assert inputs.ndim == 4
        inputs = inputs.clone().detach().requires_grad_(True).to(device)
        outputs = model(inputs)
    if outputs_activation_func is not None:
        outputs = outputs_activation_func(outputs)
    if label is None:
        label = int(torch.argmax(outputs))
    if isinstance(label, int):
        outputs = outputs[:, label]
        grads = torch.autograd.grad(outputs.sum(), [inputs], create_graph=True)[0].unsqueeze(1)
    else:
        grads = torch.stack([torch.autograd.grad(outputs[:, l].sum(), [inputs], create_graph=True)[0]
                             for l in label], dim=1)
        outputs = outputs[:, label]
    grads = grads[0].detach()
    outputs = outputs[0].detach()
    if return_outputs:
        return grads, outputs
    else:
        return grads


def perturbatation_gradients(model: Module,
                             inputs: Tensor,
                             modality: str = 'image',
                             n: int = 1,
                             attention_mask: Tensor = None,
                             covariance: Union[int, float, np.ndarray, Tensor] = None,
                             cholesky_decomposed_covariance: Union[np.ndarray, Tensor] = None,
                             per_channel: bool = False,
                             var_spread: float = 0.15,
                             pertubation_device: str = None,
                             label: Union[int, List[int]] = None,
                             outputs_activation_func=None,
                             return_outputs: bool = False):
    # inputs shape should be: (channel (or tokens), features, ... features) where features will be flattened
    # covariance should be flattened.
    grads = []
    outputs = []
    for i in range(n):
        perturbation = perturbate(inputs, covariance, cholesky_decomposed_covariance,
                                  per_channel, var_spread, pertubation_device)
        grad, output = extract_gradients(model, perturbation, modality, attention_mask,
                                         label, outputs_activation_func,
                                         return_outputs=True)
        grads.append(grad)
        outputs.append(output)
    if return_outputs:
        return grads, outputs
    else:
        return grads


def smooth_grad(grads: List[Tensor],
                sum_grad_dim: int = None):
    grads = torch.stack(grads, dim=-1)
    if sum_grad_dim:
        grads = grads.sum(dim=sum_grad_dim)
    return grads.mean(dim=-1)


def smooth_grad_sq(grads: List[Tensor],
                   sum_grad_dim: int = None):
    grads = torch.stack(grads, dim=-1)
    if sum_grad_dim:
        grads = grads.sum(dim=sum_grad_dim)
    return (grads ** 2).mean(dim=-1)


def var_grad(grads: List[Tensor],
             sum_grad_dim: int = None):
    grads = torch.stack(grads, dim=-1)
    if sum_grad_dim:
        grads = grads.sum(dim=sum_grad_dim)
    return (grads ** 2).var(dim=-1)


def functional_information(grads: List[Tensor],
                           outputs: List[Tensor],
                           sum_grad_dim: int = None):
    grads = torch.stack(grads, dim=-1)
    outputs = torch.stack(outputs, dim=-1)
    # grads.shape = (labels, channels, features..., n)
    # outputs,shape = (labels, n)
    if sum_grad_dim:
        grads = grads.sum(dim=sum_grad_dim)
    outputs = broadcast_inner_dims(outputs, grads)
    grads = (grads ** 2) / outputs
    return grads.mean(dim=-1)


def covariance_functional_information(grads: List[Tensor],
                                      outputs: List[Tensor],
                                      covariance: Tensor,
                                      sum_grad_dim: int = None,
                                      return_cov_grads: bool = False,
                                      device: str = None):
    if device is None:
        device = covariance.device
    covariance = covariance.to(device)
    grads = torch.stack(grads, dim=-1).to(device)
    outputs = torch.stack(outputs, dim=-1).to(device)
    # grads.shape = (labels, channels, features..., n)
    # outputs,shape = (labels, n)
    if isinstance(covariance, np.ndarray):
        covariance = torch.tensor(covariance).to(grads.device)
    # the covariance is a scalar
    if isinstance(covariance, (int, float)) or covariance.ndim < 1:
        cov_grads = covariance * grads
    # the covariance is a vector (a scalar for each channel)
    # grads.shape = (labels, channels, features..., n)
    # covariance.shape = (channels,)
    elif covariance.ndim == 1 and covariance.shape[0] == grads.shape[0]:
        cov_grads = broadcast_trailing_dims(covariance.unsqueeze(0), grads) * grads
    else:
        # grads.shape = (labels, channels, features..., n)
        grads = torch.movedim(grads, -1, 1)
        # grads.shape = (labels, n, channels, features...)
        grads_shape = list(grads.shape)
        grads = flatten(grads, ndims_keep=3)
        # grads.shape = (labels, n, channels,flatten_features...)
        # a covariance matrix
        if covariance.ndim == 2:
            cov_grads = torch.einsum("ij, lncj -> lnci", covariance, grads)
        # a tensor, first dim is channels/tokens
        else:
            cov_grads = torch.einsum("cij, lncj -> lnci", covariance, grads)
        grads = torch.movedim(grads.reshape(*grads_shape), 1, -1)
        cov_grads = torch.movedim(cov_grads.reshape(*grads_shape), 1, -1)
    if sum_grad_dim:
        cov_grads = cov_grads.sum(dim=sum_grad_dim)
        grads = grads.sum(dim=sum_grad_dim)
    if return_cov_grads:
        return cov_grads.mean(dim=-1)
    grads = cov_grads * grads
    outputs = broadcast_inner_dims(outputs, grads)
    grads = grads / outputs
    return grads.mean(dim=-1)


def explain(model: Module,
            inputs: Tensor,
            method: Union[str, List[str]] = 'functional_information',
            modality: str = 'image',
            n: int = 1,
            attention_mask: Tensor = None,
            covariance: Union[int, float, np.ndarray, Tensor] = None,
            cholesky_decomposed_covariance: Union[np.ndarray, Tensor] = None,
            per_channel: bool = False,
            var_spread: float = 0.15,
            pertubation_device: str = None,
            label: Union[int, List[int]] = None,
            outputs_activation_func=None):
    """

    :param model: the model to explain
    :param inputs: the inputs to the model, should be a tensor of shape (channels/token, features, ...)
    :param method: the method to use to explain the model,
     can be 'functional_information', 'smooth_grad', 'smooth_grad_sq', 'var_grad', 'covariance_functional_information'
     or 'covariance_gradients' (which is only the multiplication of the covariance and the gradients).
     It can also be a list of these methods (in that case it returns a dict with the method name as key and
      the explanation as value).
    :param modality: the modality of the inputs, can be 'image', 'text', 'audio'
    :param n: the number of perturbations to generate
    :param attention_mask: the attention mask to use for the model if it is a PreTrainedModel
    :param covariance: the covariance matrix to use for the `covariance_functional_information` method.
            Can be either None (then estimate the variance over each channel/token), a scalar,
            a vector (of the length of the number of channels), a covariance matrix, or a tensor (in that case,
            the first dim is the channel/token dim)
    :param cholesky_decomposed_covariance: the cholesky decomposition of the `covariance` matrix.
            If None, it will be computed
    :param per_channel: if True, the estimated variance (in case `covariance` is None)
                         will be computed per channel/token.
    :param var_spread: the spread of the variance to use (multiplying the covariance by it).
    :param pertubation_device: the device to use for the perturbations. If None, it will be the same as the model.
    :param label: the label to explain. If None, the prediction will be used as the label.
    :param outputs_activation_func: the activation function to use for the outputs.
                                     If None, the identity function will be used.
    :return: an explanation - a tensor of shape (channels/token, features, ...),
             or a dict of explanation tensors if `method` is a list of methods.
    """
    if isinstance(method, str):
        method = [method]
    if modality == 'image':  # sum over the channels, the explanation shape is (features, ...)
        sum_grad_dim = 1
    elif modality == 'text':  # sum over the tokens, the explanation shape is (tokens, ...)
        sum_grad_dim = 2
    else:
        sum_grad_dim = None
    pert_grad_kwargs = dict(model=model, inputs=inputs, modality=modality, n=n, attention_mask=attention_mask,
                            covariance=covariance, cholesky_decomposed_covariance=cholesky_decomposed_covariance,
                            per_channel=per_channel, var_spread=var_spread, pertubation_device=pertubation_device,
                            label=label, outputs_activation_func=outputs_activation_func, return_outputs=True)

    def _non_cov_method(pert_grad_kwargs, methods: List[str]):
        pert_grad_kwargs = pert_grad_kwargs.copy()
        pert_grad_kwargs.update(dict(covariance=None, cholesky_decomposed_covariance=None))
        grad, output = perturbatation_gradients(**pert_grad_kwargs)
        explanations = {}
        for method in methods:
            if method == 'smooth_grad':
                exp = smooth_grad(grad, sum_grad_dim)
            elif method == 'smooth_grad_sq':
                exp = smooth_grad_sq(grad, sum_grad_dim)
            elif method == 'var_grad':
                exp = var_grad(grad, sum_grad_dim)
            elif method == 'functional_information':
                exp = functional_information(grad, output, sum_grad_dim)
            else:
                raise ValueError(f'`method` {method} is not supported')
            explanations[method] = exp
        return explanations

    def _cov_method(pert_grad_kwargs, methods: List[str]):
        pert_grad_kwargs = pert_grad_kwargs.copy()
        pert_grad_kwargs.update(dict(covariance=covariance, cholesky_decomposed_covariance=cholesky_decomposed_covariance))
        grad, output = perturbatation_gradients(**pert_grad_kwargs)
        explanations = {}
        for method in methods:
            if method == 'covariance_gradients':
                exp = covariance_functional_information(grad, output, covariance, sum_grad_dim,
                                                        True, pertubation_device)
            elif method == 'covariance_functional_information':
                exp = covariance_functional_information(grad, output, covariance, sum_grad_dim,
                                                        False, pertubation_device)
            else:
                raise ValueError(f'`method` {method} is not supported')
            explanations[method] = exp
        return explanations

    cov_methods = [m for m in method if m in ['covariance_gradients', 'covariance_functional_information']]
    non_cov_methods = [m for m in method if m not in ['covariance_gradients', 'covariance_functional_information']]
    explanations = {}

    if len(non_cov_methods) > 0:
        explanations.update(_non_cov_method(pert_grad_kwargs, non_cov_methods))
    if len(cov_methods) > 0:
        explanations.update(_cov_method(pert_grad_kwargs, cov_methods))

    if label is None or isinstance(label, int):
        explanations = {m: exp[0] for m, exp in explanations.items()}
    if len(method) == 1:
        explanations = explanations[method[0]]
    return explanations


def explain_batch(model: Module,
                  inputs: List[Tensor],
                  method: Union[str, List[str]] = 'functional_information',
                  modality: str = 'image',
                  n: int = 1,
                  attention_mask: List[Tensor] = None,
                  covariance: Union[int, float, np.ndarray, Tensor, List] = None,
                  cholesky_decomposed_covariance: Union[np.ndarray, Tensor, List] = None,
                  per_channel: bool = False,
                  var_spread: float = 0.15,
                  pertubation_device: str = None,
                  label: Union[int, List[int]] = None,
                  outputs_activation_func=None):
    """
    Same parameters as `explain()`, except the `inputs` and `attention_mask` should be a list of tensors.
         `covariance`/`cholesky_decomposed_covariance`/`label` can also be lists with the same length as `inputs`.
    """
    assert isinstance(inputs, list)
    assert attention_mask is None or (isinstance(attention_mask, list) and len(attention_mask) == len(inputs))

    if covariance is None or isinstance(covariance, (int, float, np.ndarray, Tensor)):
        covariance = [covariance] * len(inputs)
    else:
        assert (isinstance(covariance, list) and len(covariance) == len(inputs))

    if cholesky_decomposed_covariance is None or isinstance(cholesky_decomposed_covariance,
                                                            (int, float, np.ndarray, Tensor)):
        cholesky_decomposed_covariance = [cholesky_decomposed_covariance] * len(inputs)
    else:
        assert (isinstance(cholesky_decomposed_covariance, list) and len(cholesky_decomposed_covariance) == len(inputs))

    if label is None or isinstance(label, int):
        label = [label] * len(inputs)
    else:
        assert (isinstance(label, list) and len(label) == len(inputs))
    explanations = []
    for _inputs, _covariance, _cholesky_decomposed_covariance, _label in zip(inputs, covariance,
                                                                             cholesky_decomposed_covariance, label):
        explanations.append(explain(model, _inputs, method, modality, n, attention_mask, _covariance,
                                    _cholesky_decomposed_covariance, per_channel, var_spread, pertubation_device,
                                    _label, outputs_activation_func))
    return explanations

