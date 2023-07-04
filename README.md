[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Docs](https://github.com/mines-opt-ml/spo-tos/actions/workflows/ci.yml/badge.svg)

# Faster Predict-and-Optimize (FPO) with Davis-Yin Splitting (DYS)

In many practical settings, a combinatorial problem must be repeatedly solved with similar, but distinct parameters. Yet, the parameters $w$ are not directly observed; only contextual data $d$ that correlates with $w$ is available. It is tempting to use a neural network to predict $w$ given $d$, but training such a model requires reconciling the discrete nature of combinatorial optimization with the gradient-based frameworks used to train neural networks. One approach to overcoming this issue is to consider a continuous relaxation of the combinatorial problem. While existing such approaches have shown to be highly effective on small problems (10--100 variables) they do not scale well to large problems. In this work, we show how recent results in operator splitting can be used to design such a system which is easy to train  and scales effortlessly to problems with thousands of variables.

## Publication

_Faster Predict-then-Optimize with Davis-Yin Splitting_ (**[arXiv Link](https://arxiv.org/abs/2301.13395)**)
    
    @article{mckenzie2023faster,
      title={Faster Predict-and-Optimize with Davis-Yin Splitting},
      author={McKenzie, Daniel and Wu Fung, Samy and Heaton, Howard},
      journal={arXiv preprint arXiv:2301.13395},
      year={2023}
    }

See [documentation site](https://fpo-tos.research.typal.academy) for more details.
