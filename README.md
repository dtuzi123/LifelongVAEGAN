# LifelongVAEGAN

This is the implementation of the Lifelong VAEGAN

# Title : Learning latent representations across multiple data domains using Lifelong VAEGAN

# Paper link

https://link.springer.com/chapter/10.1007/978-3-030-58565-5_46

# Abstract

The problem of catastrophic forgetting occurs in deep learning models trained on multiple databases in a sequential manner. Recently, generative replay mechanisms (GRM) have been proposed to reproduce previously learned knowledge aiming to reduce the forgetting. However, such approaches lack an appropriate inference model and therefore can not provide latent representations of data. In this paper, we propose a novel lifelong learning approach, namely the Lifelong VAEGAN (L-VAEGAN), which not only induces a powerful generative replay network but also learns meaningful latent representations, benefiting representation learning. L-VAEGAN can allow to automatically embed the information associated with different domains into several clusters in the latent space, while also capturing semantically meaningful shared latent variables, across different data domains. The proposed model supports many downstream tasks that traditional generative replay methods can not, including interpolation and inference across different
data domains.

# Environment

1. Tensorflow 1.5
2. Python 3.6

# BibTex

@inproceedings{ye2020learning,
  title={Learning latent representations across multiple data domains using Lifelong VAEGAN},
  author={Ye, Fei and Bors, Adrian G},
  booktitle={European Conference on Computer Vision},
  pages={777--795},
  year={2020},
  organization={Springer}
}
