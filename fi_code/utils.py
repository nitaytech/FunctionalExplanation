import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.transforms.functional import resize
from typing import List, Tuple, Union, Any, Dict
from matplotlib import pyplot as plt
from matplotlib.pyplot import Axes
import html
import os
import json
import pickle


def to_path(path: Union[str, Path], make_parent_dir: bool = False):
    if isinstance(path, str):
        path = Path(path)
    if make_parent_dir:
        os.makedirs(path.parent, exist_ok=True)
    return path


def save_pkl(to_save: Any, file_path: Union[str, Path]):
    file_path = to_path(file_path, make_parent_dir=True)
    with open(file_path, 'wb') as f:
        pickle.dump(to_save, f)


def load_pkl(file_path: Union[str, Path]):
    file_path = to_path(file_path)
    with open(file_path, 'rb') as f:
        instance = pickle.load(f)
        return instance


def save_json(to_save: Any, file_path: Union[str, Path]):
    file_path = to_path(file_path, make_parent_dir=True)
    with open(file_path, 'w') as f:
        json.dump(to_save, f)


def load_json(file_path: Union[str, Path]):
    file_path = to_path(file_path)
    with open(file_path, 'r') as f:
        instance = json.load(f)
        return instance


def divide_to_batches(list_to_divide: List, batch_size: int):
    return [list_to_divide[i * batch_size: (i + 1) * batch_size]
            for i in range(int(np.ceil(len(list_to_divide) / batch_size)))]


def s_curve(x: np.ndarray, k: float = -0.5):
    return ((k - 1) * (2 * x - 1)) / (2 * (4 * k * np.abs(x - 0.5) - k - 1)) + 0.5


def standard(x: np.ndarray):
    return (x - x.mean()) / x.std()


def min_max_scaler(x: np.ndarray):
    vmin, vmax = np.min(x), np.max(x)
    return (x - vmin) / (vmax - vmin)


def get_indices(dataset: Dataset, indices: List[int] = None):
    if indices is None:
        subset = dataset
    else:
        subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=len(subset))
    return next(iter(loader))


def save_image(img: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def resize_img(image, target_size: Tuple[int, int] = (128, 128)):
    image_new = resize(image, list(target_size))
    return image_new


def preprocess_image(img: np.ndarray, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=list(mean), std=list(std))
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def get_mpl_colormap(cmap_name: str = 'bwr'):
    cmap = plt.get_cmap(cmap_name)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    return color_range.reshape(256, 1, 3)


def show_explanation(img: Union[np.ndarray, Tensor, None],
                     explanation: Union[np.ndarray, Tensor],
                     img_format: str = 'bgr',
                     alpha_values: Tuple[float, float] = None,
                     colormap: str = 'bwr_r',
                     k: float = -0.5,
                     label: str = None,
                     ax: Axes = None) -> Union[Axes, np.ndarray]:
    # img.shape = (height, width, channels)
    # explanation.shape = (height, width)
    if isinstance(img, Tensor):
        img = img.numpy()
    if isinstance(explanation, Tensor):
        explanation = explanation.detach().cpu().numpy()
    explanation = s_curve(min_max_scaler(explanation), k)  # adding the channel dim, which is the last
    heatmap = cv2.applyColorMap(np.uint8(255 * explanation), get_mpl_colormap(colormap))
    heatmap = np.float32(heatmap) / 255
    if img is None:
        alpha_weights = 1.0
        img = 0.0
    else:
        if img_format == 'gray':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img_format == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.float32(img) / 255
        alpha_weights = min_max_scaler(np.abs(explanation - np.median(explanation)))
        if alpha_values is not None:
            avmin, avmax = np.quantile(explanation, alpha_values[0]), np.quantile(explanation, alpha_values[1])
            alpha_weights[((avmin <= explanation) & (explanation <= avmax))] = 0
        alpha_weights = np.expand_dims(alpha_weights, -1)
    exp = alpha_weights * heatmap + (1 - alpha_weights) * img
    exp = exp / np.max(exp)
    exp = np.uint8(255 * exp)
    if ax is None:
        return exp
    if label is not None:
        ax.set_title(label)
    ax.imshow(exp)
    ax.axis('off')
    return ax


def show_explanations(imgs: Union[List[Union[np.ndarray, Tensor, None]], np.ndarray, Tensor, None],
                      explanations: Union[List[Union[np.ndarray, Tensor]], np.ndarray, Tensor],
                      img_format: str = 'bgr',
                      alpha_values: Tuple[float, float] = None,
                      colormap: str = 'bwr_r',
                      labels: Union[str, List[str], None] = None,
                      k: float = -0.5,
                      figsize: Tuple[int, int] = (12, 5),
                      n_rows=None,
                      n_cols: int = None):
    if not isinstance(imgs, list) and not isinstance(explanations, list):
        imgs, explanations = [imgs], [explanations]
    elif isinstance(imgs, list) and not isinstance(explanations, list):
        explanations = [explanations] * len(imgs)
    elif not isinstance(imgs, list) and isinstance(explanations, list):
        imgs = [imgs] * len(explanations)
    if not isinstance(labels, list):
        labels = [None] * len(imgs)
    if n_rows is None and n_cols is None:
        n_rows = int(len(imgs) ** 0.5)
        n_cols = int(np.ceil(len(imgs) / n_rows))
    elif n_rows is None and n_cols is not None:
        n_rows = int(np.ceil(len(imgs) / n_cols))
    elif n_cols is None:
        n_cols = int(np.ceil(len(imgs) / n_rows))
    _, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_cols * n_rows == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    for img, explanation, label, ax in zip(imgs, explanations, labels, axs[:len(imgs)]):
        show_explanation(img, explanation, img_format=img_format, alpha_values=alpha_values,
                         colormap=colormap, label=label, k=k, ax=ax)
    for ax in axs[len(imgs):]:
        ax.axis('off')
    plt.show()


def show_explanations_grid(imgs: List[Union[np.ndarray, Tensor]],
                           explanations: List[Union[np.ndarray, Tensor, Dict[str, Union[np.ndarray, Tensor]]]],
                           labels: List[str] = None,
                           figsize: Tuple[Union[int, float], Union[int, float]] = (15, 7.5),
                           colormap: str = 'Greys_r',
                           k: float = -0.5):
    assert len(imgs) == len(explanations)
    assert labels is None or len(labels) == len(imgs)
    n_cols = len(imgs)
    if isinstance(explanations[0], dict):
        n_rows = len(explanations[0])+1
        methods = list(explanations[0].keys())
    else:
        n_rows = 2
        methods = []
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    subfigs = fig.subfigures(nrows=n_rows, ncols=1)
    for row, subfig in enumerate(subfigs):
        if row == 0:
            # plotting images and labels
            axs = subfig.subplots(nrows=1, ncols=n_cols)
            for col, ax in enumerate(axs):
                img = imgs[col]
                if isinstance(img, Tensor):
                    img = img.detach().cpu().numpy()
                ax.imshow(np.moveaxis(img, 0, -1))
                if labels is not None:
                    ax.set_title(labels[col])
                ax.axis('off')
        else:
            # plotting explanations
            if isinstance(explanations[0], dict):
                method = methods[row-1]
                subfig.suptitle(method)
            else:
                method = None
            axs = subfig.subplots(nrows=1, ncols=n_cols)
            for col, ax in enumerate(axs):
                if method is not None:
                    explanation = explanations[col][method]
                else:
                    explanation = explanations[col]
                if isinstance(explanation, Tensor):
                    explanation = explanation.detach().cpu().numpy()
                explanation = s_curve(min_max_scaler(explanation), k=k)
                ax.imshow(explanation, cmap=colormap)
                ax.axis('off')
    plt.show()


def textual_explanation_to_html(tokens: List[str],
                                explanation: Union[np.ndarray, Tensor],
                                colormap: str = 'bwr',
                                label: str = None,
                                no_space_prefix: str = '##',
                                k: float = -0.5) -> str:
    # explanation.shape = (tokens, )
    if isinstance(explanation, Tensor):
        explanation = explanation.detach().cpu().numpy()
    cmap = plt.get_cmap(colormap)
    html_code = ''
    if label is not None:
        html_code += f'<b>{label}: </b>'
    explanation = (s_curve(min_max_scaler(explanation), k) * 0.5).tolist()
    for token, weight in zip(tokens, explanation):
        color = [int(255 * c) for c in cmap(weight)]
        if not token.startswith(no_space_prefix):
            html_code += ' '
        html_code += f'<span style="background-color:rgba{tuple(color)};">{html.escape(token)}</span>'
    return html_code


def textual_explanations_to_html(tokens: Union[List[List[str]], List[str]],
                                 explanations: List[Union[np.ndarray, Tensor, Dict[str, Union[np.ndarray, Tensor]]]],
                                 colormap: str = 'bwr',
                                 labels: List[str] = None,
                                 no_space_prefix: str = '##',
                                 k: float = -0.5) -> str:
    if isinstance(tokens[0], str) and not isinstance(explanations, list):
        tokens, explanations = [tokens], [explanations]
    elif isinstance(tokens[0], str) and isinstance(explanations, list):
        tokens = [tokens] * len(explanations)
    if not isinstance(labels, list):
        labels = [None] * len(tokens)
    html_code = ''
    for toks, exp, label in zip(tokens, explanations, labels):
        if isinstance(exp, dict):
            for method, explanation in exp.items():
                mlabel = f"{method}, {label}"
                html_code += textual_explanation_to_html(toks, explanation, colormap, mlabel, no_space_prefix)
                html_code += '<br>'
        else:
            html_code += textual_explanation_to_html(toks, exp, colormap, label, no_space_prefix)
        html_code += '<hr>'
    return html_code
