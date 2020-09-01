[TOC]

# adVAE: a Self-adversarial Variational Autoencoder with Gaussian Anomaly Prior Knowledge for Anomaly Detection

Xu hong Wang,∗, Ying Du, Shijie Lin, Ping Cui, Yuntian Shen, Yupu Yanga

Shanghai Jiao Tong University, Shanghai,

China Wu han University, Wuhan,

China University of California, Davis , U.S.A.

----

## Abstract

Recently, deep generative models have become increasingly popular in unsupervised anomaly detection. However,deep generative models aim at recovering the data distribution  rather  than detecting anomalies. Moreover, deep generative models have the risk of overfitting training samples, which has disastrous effects on anomaly detection performance. To solve the above two problems, we propose a self-adversarial variational autoencoder (adVAE) with a Gaussian anomaly prior assumption. We assume that both the anomalous and the normal prior distribution are Gaussian and have overlaps in the latent space. Therefore, a Gaussian transformer net T is trained to synthesize anomalousbut near-normal latent variables. Keeping the original training objective of a variational autoencoder, a generator G tries to distinguish between the normal latent variables encoded by E and the anomalous latent variables synthesized by T, and the encoder E is trained to discriminate whether the output of G is real. These new objectives we added not only give both G and E the ability to discriminate, but also become an additional regularization mechanism to prevent overfitting. Compared with other competitive methods, the proposed model achieves significant improvements in extensive experiments. The employed datasets and our model are available in a Github repository.

最近，深度生成模型在无间督异常检测领域变得越来越流行。然而，深度生成模型旨在恢复数据分布而非检测异常.此外,深度生成模型还存在过拟合样本的风险,这会极大影响其异常检测性能.为了解决以上两个问题,本文提出了带有高斯异常先验假设的自对抗的变分自编码器.我们假设异常和正常样本的先验分布都是服从高斯分布且在隐空间有重叠区域.基于此,本文训练一个高斯变换网络T来合成接近正常隐向量的异常.然后保留变分自编码的原始训练目的,使用一个生成器G来分辨由编码器E产生的正常隐向量和由T合成的异常隐向量,同时编码器E还负责分辨G的输出是否真实.本文添加的这些新目的不仅使得G和E由了分辨能力,同时还成为了一种额外的正则策略来抑制了过拟合.和其他方法相比,本文模型在大量的实验中取得了显著的进步.实验使用的数据集和模型可以在Github取得.

*Keywords*:anomaly detection, outlier detection, novelty detection, deep generative model, variational autoencoder

关键词: 异常检测,离群点检测,新颖性检测,深度生成模型,变分自编码器

----

1.Introduction

Anomaly detection (or outlier detection) can be regarded as the task of identifying rare data items that differ from the majority of the data. Anomaly detection is applicable in a variety of domains, such as intrusion detection, fraud detection, fault detection,health monitoring, and security checking [1–5].Owing to the lack of labeled anomaly samples, there is alarge skew between normal and anomaly class distributions. Some attempts [6–8] use several imbalanced-learning methods to improve the performance of supervised anomaly detection models. Moreover, unsupervised models are more popular than supervised models in the anomaly detection field. Reference [9] reviewed machine-learning-based anomaly detection algorithms comprehensively.

异常检测(离群点检测)可以被视为一个区分罕见数据和常见数据区别的任务.异常检测在入侵检测,欺诈检测,故障检测,健康监测和安全检查等诸多领域都有应用[2-5].由于缺少标记的异常样本,异常样本和正常样本的分布有很大差异.一些工作[6-8]尝试使用不平衡的学习方法来提升异常检测监督模型的性能.然而,无监督模型比监督模型在异常检测领域要更受欢迎.文献[9]非常全面的总结了各类基于机器学习的异常检测算法.

Recently, deep generative models have become increasingly popular in anomaly detection [10]. A generative model can learn a probability distribution model by being trained on an anomaly-free dataset. Afterwards, outliers can be detected by their deviation from the probability model. The most famous deep generative models are variational autoencoders (VAEs) [11]and generative adversarial networks (GANs) [12].

最近,深度生成模型在异常检测领域变得越来越流行[10].深度生成模型可以在正常样本上学习到数据的概率分布.之后,通过衡量概率模型的偏离值就可以检测离群点.变分自编码器(VAE)和GAN是最出名的深度生成模型.

VAEs have been used in many anomaly detection studies. The first work using VAEs for anomaly detection [13] declared that VAEs generalize more easily than autoencoders (AEs), because VAEs work on probabilities. [14, 15] used different types of RNN-VAE architectures to recognize the outliers of time series data. [1]and [16] implemented VAEs for intrusion detection and internet server monitoring, respectively. Furthermore,GANs and adversarial autoencoders [17] have also been introduced into image [4, 18] and video [19] anomaly detection.

在很多异常检测研究里都用到了VAE.文献[13]是第一个在异常检测上使用VAE的工作,它揭示了由于VAE是在概率分布上工作的,因此VAE的生成能力比AE要好.[14,15]使用了不同结构的RNN-VAE来识别时间序列上的离群点.[1]和[16]则将VAE应用于入侵检测和互联网服务器监测.此外,GAN和对抗AE也被用于图像和视频的异常检测.

However, there are two serious problems in deep-generative-model-based anomaly detection methods.

然而,基于深度生成模型的异常检测方法依然存在两个严重问题.

(1) Deep generative models only aim at recovering the data distribution of the training set, which has an indirect contribution to detecting anomalies. Those earlier studies paid little attention to customize their models for anomaly detection tasks. Consequently, there is an enormous problem that those models only learnfrom available normal samples, without attempting todiscriminate the anomalous data. Owing to the lack of discrimination, it is hard for such models to learn useful deep representations for anomaly detection tasks.

(1)深度生成模型主要是尝试恢复训练数据的数据分布,以此来间接实现异常检测.之前的工作都很少关注针对异常检测任务进行模型的改进.这导致这些模型注重从可用的正常样本中学习分布而没有试图判别异常数据.由于缺少判别性,这些模型难以学习到对异常检测任务有用的深度表征.

(2) Plain VAEs use the regularization of the Kullback–Leibler divergence (KLD) to limit the capacity of the encoder, but there is no regularization implemented in the generator. Because neural networksare universal function approximators, generators can, intheory, cover any probability distribution, even withoutdependence on the latent variables [20]. However, previous attempts [21, 22] have found it hard to benefit from using an expressive and powerful generator. [20]uses Bits-Back Coding theory [23] to explain this phenomenon: if the generator can model the data distribution $p_{data}(x)$ without using information from the latent code $z$, it is more inclined not to use $z$. In this case, to reduce the KLD cost, the encoder loses much information about $x$ and maps $p_{data}(x)$ to the simple prior $p(z)$ (e.g.,$N(0,I)$) rather than the true posterior $p(z|x)$. Once this undesirable training phenomenon occurs, VAEs tend to overfit the distribution of existing normal data, which leads to a bad result (e.g., a high false positive rate),especially when the data are sparse. Therefore, the capacity of VAEs should be limited by an additional regularization in anomaly detection tasks.

(2)常规VAE使用KL散度来限制编码器,但是对于生成器却没有正则策略.由于神经网络是通用的函数拟合器,那么理论上生成器可以不依赖隐变量而覆盖任何概率分布[20].然而[21,22]发现即使使用能力强大的生成器也收效甚微.文献[20]使用Bits-Back编码理论[23]解释了这种现象:若生成器可以在不使用隐变量编码$z$的情况下也可以对数据$P_{data}(x)$进行建模,那么生成器将更加倾向不使用$z$来进行建模.在这种情况下,减小KL散度的损失,编码器将丢失更多关于$x$的信息,且将分布$p_{data}(x)$映射到简单的先验分布$p(z)$而非后验分布$p(z|x)$.一旦这种不良的训练情况发生,VAE将倾向于过拟合现存的正常数据分布,这将导致模型出现很高的假阳率,特别是在数据稀疏时.因此需要额外的正则策略来限制VAE的能力.

There are only a handful of studies that attempt to solve the above two problems. MO-GAAL method [24]made the generator of a GAN stop updating before convergence in the training process and used the nonconverged generator to synthesize outliers. Hence, the discriminator can be trained to recognize outliers in a supervised manner. To try to solve the mode collapse issue of GANs, [24] expands the network structure from a single generator to multiple generators with different objectives. [25] proposed an assumption that the anomaly prior distribution is the complementary set of the normal prior distribution in latent space. However, we believe that this assumption may not hold true. If the anomalous and the normal data have complementary distributions, which means that they are separated in the latent space,then we can use a simple method (such as KNN) to detect anomalies and achieve satisfactory results, but this is not the case. Both normal data and outliers are generated by some natural pattern. Natural data ought to conform to a common data distribution, and it is hard to imagine a natural pattern that produces such a strange distribution.

只有少数研究尝试解决以上两个问题.MO-GAAL方法[24]在训练收敛前停止来生成器GAN,并使用未收敛的生成器来合成离群点.这样判别器可以以监督学习的方式来训练识别离群点.另外为了解决GAN模式崩溃的问题,[24]还将网络结构从单一的生成器扩展成多个不同目标的生成器.[25]则假设异常先验分布是正常先验分布在隐空间中的补集.但是,我们认为这个假设可能不成立.若异常和正常数据是互补分布,那么这意味着他们在隐空间中是分离的,那么我们是使用KNN等方法就可以较好的检测出异常样本,但显然这是不可能的.正常数据和离群点都是由某个自然分布产生的,自然数据应该符合一个通用的数据分布,并且很难想象是怎样的一个自然模式会产生这样一个奇怪的分布.

To enhance deep generative models to distinguish between normal and anomalous samples and to prevent them from overfitting the given normal data, we propose a self-adversarial Variational Autoencoder (ad-VAE) with a Gaussian anomaly prior assumption and a self-adversarial regularization mechanism. The basic idea of this self-adversarial mechanism is adding discrimination training objectives to the encoder and the generator through adversarial training. These additional objectives will solve the above two problems at the same time; the details are as follows.

为了加强深度生成模型区分正常和异常样本的能力,同时防止模型在正常数据上过拟合,我们提出了一个具有高斯异常先验假设和自对抗正则机制的自对抗VAE.这个自对抗机制的大致思想是通过对抗训练,给编码器和生成器添加判别训练数据的目标.这些额外的目标将同时解决上述两个问题;详见后文.

The encoder can be trained to discriminate the original sample and its reconstruction, but we do not have any anomaly latent code to train the generator. To synthesize the anomaly latent code, we propose a Gaussian anomaly hypothesis to describe the relationship between normal and anomaly latent space. Our assumption is described in Figure 1(a); both the anomalous and the normal prior distributions are Gaussian and have overlaps in the latent space. It is an extraordinarily weak and reasonable hypothesis, because the Gaussian distri-bution is the most widespread in nature. The basic struc-ture of our self-adversarial mechanism is shown in Fig-ure 1(b). The encoder is trained to discriminate the original sample $x$ and its reconstruction $x_r$, and the generator tries to distinguish between the normal latent variables $z$ encoded by the encoder and the anomalous ones $z_T$ synthesized by $T$. These new objectives we added not only give G and E the ability to discern, but also introduce anadditional regularization to prevent model overfitting.

编码器可以被训练成可以判别原始数据和重建数据,但我们没有任何的异常隐编码来训练生成器.为了合成异常隐编码,我们提出来高斯异常假说来描述正常和异常隐空间的关系.Fig.1(a)描述我们的假设;异常和正常先验分布都是高斯分布且在隐空间上由重叠.这是一个脆弱但是合理的假设,因为高斯分布在自然界是普遍存在的.Fig.1(b)展示来本文自对抗机制的基本结构.编码器被训练来判别原始样本$x$和重建样本$x_r$,生成器则尝试区别由编码器编码的正常隐向量$z$和由$T$合成的异常样本$z_T$.这些被添加的新目标不仅使G和E有了判别能力,还引入来额外的正则化机制来防止模型过拟合.

>![Fig 1](https://raw.githubusercontent.com/hqabcxyxz/MarkDownPics/master/image/20200901183009.png)
Figure 1: (a) Our assumption is that the normal data prior distribution is a Gaussian distribution close to $N(0,I)$, and the anomalous prior is another Gaussian distribution, whose mean and variance are unknown and different from the normal prior, with overlaps in the latent space. (b) This figure illustrates the basic structure of our self-adversarial mechanism. We propose additional discrimination objectives for both the encoder and the generator by adding two competitive relationships.$x$ and $z$ are normal items because of the anomaly-free training dataset.$x_r$ and $z_T$ can be regarded as anomalous item.G is trained to distinguish $z$ and $z_T$, and E tries to discern $x$ and $x_r$.
图1:(a) 我们假设正常样本先验分布是一个接近$N(0,1)$的高斯分布,异常先验是另外一个高斯分布,其期望和方差未知且都和正常先验分布不一样,并且两者在隐空间中有重叠区域. (b) 本图展示来本文自对抗机制的大致结构.通过添加两个竞争关系,为编码器和生成器设定来额外的判别目标.$x$和$z$是正常样本.$x_r$和$z_T$被视为异常样本.G被训练来区分$z$和$z_T$,E尝试区分$x$和$x_r$

Our training process can be divided into two steps:
(1)T tries to mislead the generator G; meanwhile G works as a discriminator.
(2)G generates realistic-like samples, and the encoder E acts as a discriminator of G.
To make the training phase more robust, inspired by [26], we train alternatively between the above two steps in a mini-batch iteration.

本文训练过程可以被分为两步:
(1)T尝试误导生成器G;此时G作为一个判别器.
(2)G产生类似真实的样本,编码器E作为G的判别器.
为了使训练过程更加鲁棒,受[26]启发,在一个batch迭代中,我们交替执行以上两个步骤.

Our main contributions are summarized as follows:
1. We propose a novel and important concept that deep generative models should be customized to learn to discriminate outliers rather than being used in anomaly detection directly without any suitable customization.
2. We propose a novel self-adversarial mechanism,which is the prospective customization of a plain VAE, enabling both the encoder and the generator to discriminate outliers.
3. The proposed self-adversarial mechanism also provides a plain VAE with a novel regularization,which can significantly help VAEs to avoid over-fitting normal data.
4. We propose a Gaussian anomaly prior knowledge assumption that describes the data distribution of anomalous latent variables. Moreover, we propose a Gaussian transformer net T to integrate this prior knowledge into deep generative models.
5. The decision thresholds are automatically learned from normal data by a kernel density estimation(KDE) technique, whereas earlier anomaly detection works (e.g., [13, 16]) often omit the importance of automatic learning thresholds.

综上,本文主要贡献如下:
1. 本文提出一个新颖而重要的概念:深度生成模型应该被制定学区别离群点而不是直接被用于异常检测.
2. 本文提出一个新颖的自对抗机制:通过设计一个朴素VAE,使得编码器和生成器可以判别离群点.
3. 提出的自对抗机制为这个朴素VAE提供了一个新颖的正则化,使得VAE可以避免在正常数据上过拟合.
4. 本文提出来一个高斯异常先验知识假说来描述异常隐向量的数据分布.此外,我们提出来一个高斯变换网络T来将这些先验知识融合到深度生成模型中.
5. 决策的阈值是通过核密度估计(KDE)的方法来自动计算出来的,之前的工作常常忽略来自动学习阈值的重要性.





