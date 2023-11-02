# Faster Predict-and-Optimize (FPO) with Davis-Yin Splitting (DYS)

In many applications, a combinatorial problem must be re-
peatedly solved with similar, but distinct parameters. Yet, the
parameters w are not directly observed; only contextual data
d that correlates with w is available. It is tempting to use a
neural network to predict w given d, but training such a model
requires reconciling the discrete nature of combinatorial op-
timization with the gradient-based frameworks used to train
neural networks. When the problem in question is an Integer
Linear Program (ILP), one approach to overcoming this issue
is to consider a continuous relaxation of the combinatorial
problem. While existing methods utilizing this approach have
shown to be highly effective on small problems, they do not
scale well to large problems. In this work, we draw on ideas
from modern convex optimization to design a network and
training scheme which scales effortlessly to problems with
thousands of variables

## Installation

See ```src/requirements.txt``` for standard required packages. In addition, the code in this repository also requires the [PyEPO](https://github.com/khalil-research/PyEPO) and [blackbox-backprop](https://github.com/martius-lab/blackbox-backprop) packages.


## Quick Start
Use ```Knapsack_Experiment.ipynb``` to replicate the results of the Knapsack experiment. Run ```main_driver_shortest_path.py``` to replicate the results of the shortest path experiment. Note this file is currently set to replicate Figure 3 (i.e. the 5x5 example).