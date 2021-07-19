## Runtime Parameters For Evaluating RVFL Networks

### Model

The ```model``` parameter determines what features are passed to the network classifier.

You must pass one of the below values to ```model``` at runtime.

| Values |       Meaning       | Design |
| :------: | :----------------: | :------: |
|  f  | raw features | No network layers |
|  c  | conventional RVFL activations | Only hidden layer |
|  i  | intRVFL activations | Density-encoding and hidden layers |

### Classifier

The ```classifier``` parameter determines what classifier the network uses (trained on hidden layer activations).

Specify ```1``` or ```2``` after the classifier name depending on how you want the classifier to be trained.
```1``` trains the classifier using ```scipy.optimize.minimize``` (used in paper experiments).
```2``` trains the classifier using ```torch.optim```.

You must pass one of the below values to ```classifier``` at runtime, with ```1``` or ```2``` appended.
For example, you can pass ```glvq1``` to ```classifier``` but not ```glvq```.

| Values |       Meaning       | Method |
| :------: | :----------------: | :------: |
|  rlms  | Regularized Least Mean Squares | Supported by ```scipy``` and ```torch``` |
|  glvq  | [Generalized Learning Vector Quantization](https://papers.nips.cc/paper/1995/file/9c3b1830513cc3b8fc4b76635d32e692-Paper.pdf) | Supported by ```scipy``` and ```torch``` |
|  rslvq  | [Robust Soft Learning Vector Quantization](https://dl.acm.org/doi/10.1162/089976603321891819) | Supported by ```scipy``` |
|  kglvq  | kernalized GLVQ | Supported by ```scipy``` (```torch``` unstable) |

### Optimizer

The ```optimizer``` parameter determines what optimizer the network classifier uses during training.

You only have to pass a value to ```optimizer``` if you are training the network classifier using ```torch.optim```.
```scipy.optimize``` will use the ```lbfgs``` optimizer by default.

| Values |       Meaning       | Method |
| :------: | :----------------: | :------: |
|  sgd  | Stochastic Gradient Descent | Supported by ```torch``` |
|  adam  | Adam | Supported by ```torch``` |
|  lbfgs  | Limited-memory BFGS | Supported by ```scipy``` and ```torch```|

Note that SGD and Adam have optimization parameters such as learning rate, weight decay, etc. that can be adjusted by the user (see more [here](https://pytorch.org/docs/stable/optim.html)).
You must directly adjust them [in the program code](classifiers/lvq2.py) (in ```classifiers/lvq2.py```).

### Epochs

The ```epochs``` parameter determines the number of training epochs allowed for the network classifier.

If you are training the network classifier using ```scipy.optimize```, ```epochs``` is the maximum number of training iterations (see [here](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb)).
If you are training the classifier using ```torch.optim```, ```epochs``` is the exact number of training epochs.

The default value passed to ```epochs``` for classifiers trained using ```scipy.optimize``` is ```2500```.

### Parameter Directory

The ```param_dir``` parameter determines the network's hyperparameters. Specifically, values passed to ```param_dir``` represent the relative path of some file storing these hyperparameters.

You must pass values to ```param_dir``` in the form ```/parameters/value.csv```. The value gets parsed into a file path by appending the value to the current working directory path.


| Values |       Network       | Experiment Number |
| :------: | :----------------: | :------: |
|  ```/parameters/f_lms_param.csv``` | RLS Classifier | 1 |
|  ```/parameters/f_lvq_param.csv``` | GLVQ Classifier (1 PPC) | 1 |
|  ```/parameters/conv_lms_param.csv``` | Conventional RVFL | 2, 3 |
|  ```/parameters/conv_lvq1_param.csv``` | Conventional RVFL w/ GLVQ Classifier (1 PPC) | 2 |
|  ```/parameters/int_lms_param.csv``` | intRVFL | 3, 4, 5, 6 |
|  ```/parameters/int_lvq1_param.csv``` | intRVFL w/ GLVQ Classifier (1 PPC) | 4, 5 |
|  ```/parameters/int_lvq_param.csv``` | GLVQ-RVFL | 6, 7 |
|  ```/parameters/kglvq_param.csv``` | KGLVQ | 7 |




