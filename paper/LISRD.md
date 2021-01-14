#单应性估计 
#图像配准 

[TOC]
# Online Invariance Selection for Local Feature Descriptors
- 论文:https://arxiv.org/abs/2007.08988
- code:https://github.com/rpautrat/LISRD  
- Oral at ECCV 2020

## 摘要
是保持不变性还是保持特异性是一个难以取舍的问题.为此,本文提出了一种分解局部描述符不变性和根据上下文在线给出选择的方法.我们的框架联合学习了不同不变性的局部算子描述符和元描述符.在进行局部算子描述符匹配时,以跨图片的元描述符的的相似性为依据来确定选择正确不变性的局部算子描述符.