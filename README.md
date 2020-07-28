# Awesome Curriculum Learning[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
Some bravo or inspiring research works on the topic of curriculum learning.

A curated list of awesome Curriculum Learning resources. Inspired by [awesome-deep-vision](https://github.com/kjw0612/awesome-deep-vision), [awesome-adversarial-machine-learning](https://github.com/yenchenlin/awesome-adversarial-machine-learning), [awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers), and [awesome-architecture-search](https://github.com/markdtw/awesome-architecture-search)

#### Why Curriculum Learning?
Self-Supervised Learning has become an exciting direction in AI community. 
  - Bengio: "..." (ICML 2009)

Biological inspired learning scheme.
  - Learn the concepts by increasing complexity, in order to allow learner to exploit previously learned concepts and thus ease the abstraction of new ones.

## Contributing

Please help contribute this list by contacting [me](https://github.com/Openning07/awesome-curriculum-learning) or add [pull request](https://github.com/Openning07/awesome-curriculum-learning/pulls)

Markdown format:
```markdown
- Paper Name.
  [[pdf]](link) 
  [[code]](link)
  - Key Contribution(s)
  - Author 1, Author 2, and Author 3. *Conference Year*
```

## Table of Contents

### 2009
- Curriculum Learning.
  [[pdf]](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/15972/Bengio%2C%202009%20Curriculum%20Learning.pdf?sequence=1&isAllowed=y)
  - Rank the weights of the training examples, and use the rank to guide the order of presentation of examples to the learner.
  - Bengio, Yoshua and Louradour, J{\'e}r{\^o}me and Collobert, Ronan and Weston, Jason. *ICML 2009*

### 2015
- Self-Paced Curriculum Learning.
  [[pdf]](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14374/14349)
  - Jiang, Lu and Meng, Deyu and Zhao, Qian and Shan, Shiguang and Hauptmann, Alexander G. *AAAI 2015*

### 2017
- Self-Paced Learning: An Implicit Regularization Perspective.
  [[pdf]](https://www.researchgate.net/profile/Jian_Liang25/publication/303750070_Self-Paced_Learning_an_Implicit_Regularization_Perspective/links/5858e75b08ae3852d25555e3/Self-Paced-Learning-an-Implicit-Regularization-Perspective.pdf)
  - Fan, Yanbo and He, Ran and Liang, Jian and Hu, Bao-Gang. *AAAI 2017*
  
- Curriculum Dropout.
  [[pdf]](http://www.vision.jhu.edu/assets/MorerioICCV17.pdf) [[code]](https://github.com/pmorerio/curriculum-dropout)
  - Morerio, Pietro and Cavazza, Jacopo and Volpi, Riccardo and Vidal, Ren\'e and Murino, Vittorio. *ICCV 2017*

### 2018
- Curriculum Learning by Transfer Learning: Theory and Experiments with Deep Networks.
  [[pdf]](https://arxiv.org/pdf/1802.03796.pdf)
  - Sort the training examples based on the *performance* of a pre-trained network on a larger dataset,
    and then finetune the model to the dataset at hand.
  - Weinshall, Daphna and Cohen, Gad and Amir, Dan. *ICML 2018*

- Self-Paced Deep Learning for Weakly Supervised Object Detection.
  [[pdf]](https://arxiv.org/pdf/1605.07651.pdf)
  - Sangineto, Enver and Nabi, Moin and Culibrk, Dubravko and Sebe, Nicu. *TPAMI 2018*
  
- MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks.
  [[pdf]](http://proceedings.mlr.press/v80/jiang18c/jiang18c.pdf) [[code]](https://github.com/google/mentornet)
  - Jiang, Lu and Zhou, Zhengyuan and Leung, Thomas and Li, Li-Jia and Li, Fei-Fei. *ICML 2018*
<p align="center">
  <img src="https://github.com/google/mentornet/blob/master/images/overview.png" alt="MentorNet" width="65%">
</p>

- CurriculumNet: Weakly Supervised Learning from Large-Scale Web Images.
  [[pdf]](https://arxiv.org/pdf/1808.01097.pdf) [[code]](https://github.com/MalongTech/research-curriculumnet)
  - Guo, Sheng and Huang, Weilin and Zhang, Haozhi and Zhuang, Chenfan and Dong, Dengke and Scott, Matthew and Huang, Dinglong. *ECCV 2018*

- Unsupervised Feature Selection by Self-Paced Learning Regularization.
  [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S0167865518302782)
  - Zheng, Wei and Zhu, Xiaofeng and Wen, Guoqiu and Zhu, Yonghua and Yu, Hao and Gan, Jiangzhang. *Pattern Recognition Letters 2018*

- Curriculum Learning for Natural Answer Generation. [[pdf]](https://www.ijcai.org/Proceedings/2018/0587.pdf)
  - Cao Liu and Shizhu He and Kang Liu and Jun Zhao, *IJCAI 2018*

### 2019
- Transferable Curriculum for Weakly-Supervised Domain Adaptation
  [[pdf]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/transferable-curriculum-aaai19.pdf) [[code]](https://github.com/thuml/TCL)
  - Shu, Yang and Cao, Zhangjie and Long, Mingsheng and Wang, Jianmin. *AAAI 2019*
- Dynamic Curriculum Learning for Imbalanced Data Classification.
  [[pdf]](https://arxiv.org/pdf/1901.06783.pdf)
  - Wang, Yiru and Gan, Weihao and Wu, Wei and Yan, Junjie. *ICCV 2019*
- On The Power of Curriculum Learning in Training Deep Networks.
  [[pdf]](https://arxiv.org/pdf/1904.03626.pdf)
  - Hacohen, Guy and Weinshall, Daphna. *ICML 2019*
- Balanced Self-Paced Learning for Generative Adversarial Clustering Network.
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ghasedi_Balanced_Self-Paced_Learning_for_Generative_Adversarial_Clustering_Network_CVPR_2019_paper.pdf)
  - Ghasedi, Kamran and Wang, Xiaoqian and Deng, Cheng and Huang, Heng. *CVPR 2019*
  
- Data Parameters: A New Family of Parameters for Learning a Differentiable Curriculum. [[pdf]](http://papers.nips.cc/paper/9289-data-parameters-a-new-family-of-parameters-for-learning-a-differentiable-curriculum)
  - Saxena, Shreyas and Tuzel, Oncel and DeCoste, Dennis. *NIPS 2019*
### 2020
- BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition.
  [[pdf]](https://arxiv.org/abs/1912.02413) [[code]](https://github.com/Megvii-Nanjing/BBN)
  - Zhou, Boyan and Cui, Quan and Wei, Xiu-Shen and Chen, Zhao-Min. *CVPR 2020*
<p align="center">
  <img src="http://bbs.cvmart.net/uploads/images/202004/03/21/LsQbBcInWZ.gif?imageView2/2/w/1240/h/0" alt="BBN" width="80%">
</p>

- Uncertainty-Aware Curriculum Learning for Neural Machine Translation. [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.620/)
  
  - Zhou, Yikai and Yang, Baosong and Wong, Derek F and Wan, Yu and Chao, Lidia S. *ACL 2020*
  
- Curriculum Learning for Natural Language Understanding. [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.542/)
  
  - Benfeng, Xu and Licheng, Zhang and Zhendong, Mao and Quan, Wang and Hongtao, Xie and Yongdong, Zhang. *ACL 2020*

