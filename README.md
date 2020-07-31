#### GANbasedOversampling

Implementation of the cWGAN-based oversampling method. 
Fits a conditional Wasserstein GAN with Gradient Penalty 
and an auxiliary classifier loss to a tabular dataset with categorical and numerical attributes.
The fitted cWGAN model can than be used to resample an imbalanced training set. 
Currently only supports binary classification.

##### Implementation
Our implementation was initially based on [[1]](https://github.com/johaupt/GANbalanced/) 
and also drew upon various WGANGP pytorch implementations such as
[[2]](https://github.com/jalola/improved-wgan-pytorch) 
[[3]](https://github.com/caogang/wgan-gp) 
[[4]](https://github.com/kuc2477/pytorch-wgan-gp)
.

##### Datasets
The datasets used by our evaluation are not included in this repository but are linked to in dataloader.py. 
At the time of writing, all the datasets are publicly available.