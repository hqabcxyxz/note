[原文链接](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)

> This post explains the maths behind a generative adversarial network (GAN) model and why it is hard to be trained. Wasserstein GAN is intended to improve GANs’ training by adopting a smooth metric for measuring the distance between two probability distributions.

> 这篇文章解释了生成对抗网络(GAN)模型背后的数学原理以及为何难以训练。[[Wasserstein]] GAN旨在通过采用平滑度量来测量两个概率分布之间的距离来改善GAN的训练。

[Generative adversarial network](https://arxiv.org/pdf/1406.2661.pdf) (GAN) has shown great results in many generative tasks to replicate the real-world rich content such as images, human language, and music. It is inspired by game theory: two models, a generator and a critic, are competing with each other while making each other stronger at the same time. However, it is rather challenging to train a GAN model, as people are facing issues like training instability or failure to converge.

生成对抗网络（GAN）

