<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

Our code is implemented in [PyTorch](https://pytorch.org/), using the [Transformers](https://github.com/huggingface/transformers) libraries. 
Please make sure you have the latest version of these libraries installed.
</div>

______________________________________________________________________

# A Functional Information Perspective on Model Interpretation

### Official code repository for the ICML'2022 paper: <br> ["A Functional Information Perspective on Model Interpretation"](https://icml.cc/virtual/2022/spotlight/17196)

This work suggests a theoretical framework for model interpretability by measuring the contribution of relevant features
to the functional entropy of the network with respect to the input. 
We rely on the log-Sobolev inequality that bounds the functional entropy by the functional Fisher information
with respect to the covariance of the data.


______________________________________________________________________


## How to Run the Code

The code was tested on a Conda environment installed on Ubuntu 18.04, with a python version of 3.9.

Please make sure you have the latest version of the following libraries installed: <br>
* [PyTorch](https://pytorch.org/get-started/locally/) <br>
* [Numpy](https://www.numpy.org/get-started/quickstart/) <br>
* [OpenCV](https://opencv.org/install.html) <br>
* [Tansformers](https://huggingface.co/docs/transformers/installation) <br>

Then, clone this repo to your local machine: <br>

`git clone https://github.com/nitaytech/FunctionalExplanation.git`

Everything is ready, you can take a look at the [demo notebook](Examples.ipynb). <br>
Basically, there are two main functions which generate the explanations: 
`fi_code.fi.explain()` and `fi_code.fi.explain_batch()`, please take a look at their documentation. <br>
In addition, we also support visualizing the explanations using: `utils.show_explanations()`, 
`utils.show_explanations_grid()` and `utils.textual_explanations_to_html()`. <br>
Finally, we include a module `fi_code.basic_models.py` which contains some helper functions for training toy models
(e.g. a ResNet-based CNN and Bi/LSTM, see the notebook for some examples).




