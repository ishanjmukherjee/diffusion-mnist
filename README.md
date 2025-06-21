# Diffusion on MNIST

This is a minimal implementation of image diffusion as described in the
[DDPM](https://arxiv.org/abs/2006.11239)
paper for unconditional generation of
[MNIST](https://en.wikipedia.org/wiki/MNIST_database) images.

This repo is strongly inspired by Simo Ryu's [super minimal DDPM
implementation](https://github.com/cloneofsimo/minDiffusion/blob/master/superminddpm.py),
but I thought that taking out the time embeddings was really *too* minimal for
me.

Using a simple convnet as the underlying model, here are the images generated
after 100 epochs:

![](/assets/convnet_epoch_100.png)

On the other hand, if we use a ~similarly sized MLP, the model doesn't learn at
all, and emits random noise even after 100 epochs:

![](/assets/smolmlp_epoch_100.png)

Using a ~7x larger MLP than the convnet still outputs random noise:

![](/assets/chunkymlp_epoch_100.png)

Picking a good inductive bias matters, huh.
