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

### Mainstreams of curriculum learning

|  Tag  |        `Det`     |           `Seg`       |         `Cls`        |      `Trans`      |      `Gen`   |    Other    |
|:----------------:|:----------------:|:---------------------:|:--------------------:|:-----------------:|:------------:|:-----------:|
|  Item  | Detection | Semantic | Image Classification | Transfer Learning |  Generation  | other types |
|  Issues (e.g.,)  | long-tail | imbalance | imbalance, noise | long-tail, domain-shift |  collapose  |  -  |

### SURVEY
- Curriculum Learning: A Survey.
  [[pdf]](https://arxiv.org/pdf/2101.10382.pdf)
  - Soviany, Petru and Ionescu, Radu Tudor and Rota, Paolo and Sebe, Nicu. *arxiv 2101.10382*

### 2009
- Curriculum Learning.
  [[pdf]](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/15972/Bengio%2C%202009%20Curriculum%20Learning.pdf?sequence=1&isAllowed=y)
  - "Rank the weights of the training examples, and use the rank to guide the order of presentation of examples to the learner."
  - Bengio, Yoshua and Louradour, J{\'e}r{\^o}me and Collobert, Ronan and Weston, Jason. *ICML 2009*

### 2015
- Curriculum Learning.
  [[pdf]](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14374/14349)
  - Jiang, Lu and Meng, Deyu and Zhao, Qian and Shan, Shiguang and Hauptmann, Alexander G. *AAAI 2015*

### 2017
- Self-Paced Learning: An Implicit Regularization Perspective.
  [[pdf]](https://www.researchgate.net/profile/Jian_Liang25/publication/303750070_Self-Paced_Learning_an_Implicit_Regularization_Perspective/links/5858e75b08ae3852d25555e3/Self-Paced-Learning-an-Implicit-Regularization-Perspective.pdf)
  - Fan, Yanbo and He, Ran and Liang, Jian and Hu, Bao-Gang. *AAAI 2017*
  
- Curriculum Dropout.
  [[pdf]](http://www.vision.jhu.edu/assets/MorerioICCV17.pdf) [[code]](https://github.com/pmorerio/curriculum-dropout)
  - Morerio, Pietro and Cavazza, Jacopo and Volpi, Riccardo and Vidal, Ren\'e and Murino, Vittorio. *ICCV 2017*

- Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes.
  [[pdf]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Curriculum_Domain_Adaptation_ICCV_2017_paper.pdf) [[code]](https://github.com/YangZhang4065/AdaptationSeg)
  - Zhang, Yang and David, Philip and Gong, Boqing. *ICCV 2017*

### 2018
- Curriculum Learning by Transfer Learning: Theory and Experiments with Deep Networks.
  [[pdf]](https://arxiv.org/pdf/1802.03796.pdf)
  - "Sort the training examples based on the *performance* of a pre-trained network on a larger dataset,
    and then finetune the model to the dataset at hand."
  - Weinshall, Daphna and Cohen, Gad and Amir, Dan. *ICML 2018*

- Self-Paced Deep Learning for Weakly Supervised Object Detection.
  [[pdf]](https://arxiv.org/pdf/1605.07651.pdf)
  - Sangineto, Enver and Nabi, Moin and Culibrk, Dubravko and Sebe, Nicu. *TPAMI 2018*
  
- MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks.
  [[pdf]](http://proceedings.mlr.press/v80/jiang18c/jiang18c.pdf) [[code]](https://github.com/google/mentornet)
  - Jiang, Lu and Zhou, Zhengyuan and Leung, Thomas and Li, Li-Jia and Li, Fei-Fei. *ICML 2018*
<p align="center">
  <img src="https://github.com/google/mentornet/blob/master/images/overview.png" alt="MentorNet" width="50%">
</p>

- CurriculumNet: Weakly Supervised Learning from Large-Scale Web Images.
  [[pdf]](https://arxiv.org/pdf/1808.01097.pdf) [[code]](https://github.com/MalongTech/research-curriculumnet)
  - Guo, Sheng and Huang, Weilin and Zhang, Haozhi and Zhuang, Chenfan and Dong, Dengke and Scott, Matthew and Huang, Dinglong. *ECCV 2018*

- Unsupervised Feature Selection by Self-Paced Learning Regularization.
  [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S0167865518302782)
  - Zheng, Wei and Zhu, Xiaofeng and Wen, Guoqiu and Zhu, Yonghua and Yu, Hao and Gan, Jiangzhang. *Pattern Recognition Letters 2018*

- Learning to Teach with Dynamic Loss Functions.
  [[pdf]](https://papers.nips.cc/paper/7882-learning-to-teach-with-dynamic-loss-functions.pdf)
  - "A good teacher not only provides his/her students with qualified teaching materials (e.g., textbooks), but also sets up appropriate learning objectives (e.g., course projects and exams) considering different situations of a student."
  - Wu, Lijun Wu and Tian, Fei Tian and Xia, Yingce and Fan, Yang Fan and Qin, Tao and Lai, Jianhuang and Liu, Tie-Yan. *NeurIPS 2018*

- Progressive Growing of GANs for Improved Quality, Stability, and Variation. `Gen`
  [[pdf]](https://openreview.net/forum?id=Hk99zCeAb&noteId=Hk99zCeAb) [[code]](https://github.com/tkarras/progressive_growing_of_gans) 
  - "The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses. This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented quality."
  - Karras, Tero and Aila, Timo and Laine, Samuli and Lehtinen, Jaakko. *ICLR 2018*
<p align="center">
  <img src="https://pic1.zhimg.com/80/v2-fdaeb2fb88c40b315420b89c96460105_1440w.jpg?source=1940ef5c" alt="Progressive growing of GANs" width="55%">
</p>


### 2019
- Transferable Curriculum for Weakly-Supervised Domain Adaptation.
  [[pdf]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/transferable-curriculum-aaai19.pdf) [[code]](https://github.com/thuml/TCL)
  - Shu, Yang and Cao, Zhangjie and Long, Mingsheng and Wang, Jianmin. *AAAI 2019*

- Dynamic Curriculum Learning for Imbalanced Data Classification.
  [[pdf]](https://arxiv.org/pdf/1901.06783.pdf)[[simple demo]](https://github.com/apeterswu/L2T_loss)
  - Wang, Yiru and Gan, Weihao and Wu, Wei and Yan, Junjie. *ICCV 2019*

- Guided Curriculum Model Adaptation and Uncertainty-Aware Evaluation for Semantic Nighttime Image Segmentation.
  [[pdf]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Sakaridis_Guided_Curriculum_Model_Adaptation_and_Uncertainty-Aware_Evaluation_for_Semantic_Nighttime_ICCV_2019_paper.pdf)
  - Sakaridis, Christos and Dai, Dengxin and Gool Van Luc. *ICCV 2019*

- On The Power of Curriculum Learning in Training Deep Networks.
  [[pdf]](https://arxiv.org/pdf/1904.03626.pdf)
  - Hacohen, Guy and Weinshall, Daphna. *ICML 2019*

- Balanced Self-Paced Learning for Generative Adversarial Clustering Network.
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ghasedi_Balanced_Self-Paced_Learning_for_Generative_Adversarial_Clustering_Network_CVPR_2019_paper.pdf)
  - Ghasedi, Kamran and Wang, Xiaoqian and Deng, Cheng and Huang, Heng. *CVPR 2019*
  
### 2020
- Breaking the Curse of Space Explosion: Towards Effcient NAS with Curriculum Search.
  [[pdf]](http://proceedings.mlr.press/v119/guo20b.html) [[code]](https://github.com/guoyongcs/CNAS)
  - Guo, Yong and Chen, Yaofo and Zheng, Yin and Zhao, Peilin and Chen, Jian and Huang, Junzhou and Tan, Mingkui. *ICML 20*
<p align="center">
  <img src="https://github.com/guoyongcs/CNAS/blob/master/assets/cnas.jpg" alt="CNAS" width="45%">
</p>

- BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition.
  [[pdf]](https://arxiv.org/abs/1912.02413) [[code]](https://github.com/Megvii-Nanjing/BBN)
  - Zhou, Boyan and Cui, Quan and Wei, Xiu-Shen and Chen, Zhao-Min. *CVPR 2020*
<p align="center">
  <img src="https://pic1.zhimg.com/80/v2-de4d0bfe21a08dc1a6bbf6dae37ca998_720w.jpg" alt="BBN" width="55%">
</p>

- Open Compound Domain Adaptation.
  [[pdf]](https://arxiv.org/abs/1909.03403) [[code]](https://github.com/zhmiao/OpenCompoundDomainAdaptation-OCDA)
  - Liu, Ziwei and Miao, Zhongqi and Pan, Xingang and Zhan, Xiaohang and Lin, Dahua and Yu, Stella X and Gong, Boqing. *CVPR 2020*
<p align="center">
  <img src="https://bair.berkeley.edu/static/blog/ocda/figure_4.png" alt="OCDA" width="55%">
</p>

- Curriculum Manager for Source Selection in Multi-Source Domain Adaptation.
  [[pdf]](https://arxiv.org/pdf/2007.01261v1.pdf)[[code]](https://github.com/LoyoYang/CMSS)
  - Yang, Luyu and Balaji, Yogesh and Lim, Ser-Nam and Shrivastava, Abhinav. *ECCV 2020*
  
- Content-Consistent Matching for Domain Adaptive Semantic Segmentation. `Seg`
  [[pdf]](https://arxiv.org/pdf/2007.01261v1.pdf) [[code]](https://github.com/Solacex/CCM)
  - "to acquire those synthetic images that share similar distribution with the real ones in the target domain, so that the domain gap can be naturally alleviated by employing the content-consistent synthetic images for training."
  - "not all the source images could contribute to the improvement of adaptation performance, especially at certain training stages."
  - Li, Guangrui and Kang, Guokiang and Liu, Wu and Wei, Yunchao and Yang, Yi . *ECCV 2020*
<p align="center">
  <img src="https://pic2.zhimg.com/80/v2-f6f3eb85a79f206b4f5524eaf43a71fd_1440w.jpg" alt="CMM" width="65%">
</p>

- DA-NAS: Data Adapted Pruning for Efficient Neural Architecture Search.
  [[pdf]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720579.pdf)
  - "Our method is based on an interesting observation that the learning speed for blocks in deep neural networks is related to the difficulty of recognizing distinct categories. We carefully design a progressive data adapted pruning strategy for efficient architecture search. It will quickly trim low performed blocks on a subset of target dataset (e.g., easy classes), and then gradually find the best blocks on the whole target dataset."
  - Dai, Xiyang and Chen, Dongdong and Liu, Mengchen and Chen, Yinpeng and Yuan, Lu. *ECCV 2020*

- Label-similarity Curriculum Learning.
  [[pdf]](https://arxiv.org/pdf/1911.06902.pdf) [[code]](https://github.com/speedystream/LCL)
  - "The idea is to use a probability distribution over classes as target label, where the class probabilities reflect the similarity to the true class. Gradually, this label
representation is shifted towards the standard one-hot-encoding."
  - Dogan, Urun and Deshmukh, A. Aniket, and Machura, Marcin and Igel, Christian. *ECCV 2020*

- Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID.
  [[pdf]](https://arxiv.org/pdf/2006.02713.pdf) [[code]](https://github.com/yxgeee/SpCL) [[zhihu]](https://zhuanlan.zhihu.com/p/269112325?utm_source=wechat_session&utm_medium=social&utm_oi=41299705069568&utm_content=group3_article&utm_campaign=shareopn&wechatShare=2&s_r=0)
  - Ge, Yixiao and Zhu, Feng and Chen, Dapeng and Zhao, Rui and Li, Hongsheng. *NeurIPS 2020*

- Semi-Supervised Semantic Segmentation via Dynamic Self-Training and Class-Balanced Curriculum.
  [[pdf]](https://arxiv.org/abs/2004.08514) [[code]](https://github.com/voldemortX/DST-CBC)
  - Feng, Zhengyang and Zhou, Qianyu and Cheng, Guangliang and Tan, Xin and Shi, Jianping and Ma, Lizhuang. *arXiv 2004.08514*
  
- Evolutionary Population Curriculum for Scaling Multi-Agent Reinforcement Learning.
  [[pdf]](https://arxiv.org/pdf/2003.10423.pdf)[[code]](https://github.com/qian18long/epciclr2020)
  - "Evolutionary Population Curriculum (EPC), a curriculum learning paradigm that scales up MultiAgent Reinforcement Learning (MARL) by progressively increasing the population of training agents in a stage-wise manner."
  - Long, Qian and Zhou, Zihan and Gupta, Abhibav and Fang, Fei and Wu, Yi and Wang, Xiaolong. *ICLR 2020*

- Self-Paced Deep Reinforcement Learning.
 [[pdf]](https://papers.nips.cc/paper/2020/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html)
  - Klink, Pascal and D'Eramo, Carlo and Peters, R. Jan and Pajarinen, Joni. *NeurIPS 2020*

- Automatic Curriculum Learning through Value Disagreement.
  [[pdf]](https://papers.nips.cc/paper/2020/file/566f0ea4f6c2e947f36795c8f58ba901-Paper.pdf)
  - " When biological agents learn, there is often an organized and meaningful order to which learning happens."
  - "Our key insight is that if we can sample goals at the frontier of the set of goals that an agent is able to reach, it will provide a significantly stronger learning signal compared to randomly sampled goals"
  - Zhang, Yunzhi and Abbeel, Pieter and Pinto, Lerrel. *NeurIPS 2020*

<p align="center">
  <img src="https://github.com/Openning07/awesome-curriculum-learning/blob/master/AutomaticCurriculumLearningThroughValueDisagreement.png" alt="CMM" width="55%">
</p>

- Curriculum by Smoothing.
  [[pdf]](https://proceedings.neurips.cc/paper/2020/file/f6a673f09493afcd8b129a0bcf1cd5bc-Paper.pdf) [[code]](https://github.com/pairlab/CBS)
  - Sinha Samarth and Garg Animesh and Larochelle Hugo. *NeurIPS 2020*

- Curriculum Learning by Dynamic Instance Hardness.
  [[pdf]](https://papers.nips.cc/paper/2020/file/62000dee5a05a6a71de3a6127a68778a-Paper.pdf)
  - Zhou, Tianyi and Wang, Shengjie and Bilmes, A. Jeff. *NeurIPS 2020*

### 2021
- Robust Curriculum Learning: from clean label detection to noisy label self-correction.
  [[pdf]](https://openreview.net/pdf?id=lmTWnm3coJJ) [[online review]](https://openreview.net/forum?id=lmTWnm3coJJ)
  - "Robust curriculum learning (RoCL) improves noisy label learning by periodical transitions from supervised learning of clean labeled data to self-supervision of wrongly-labeled data, where the data are selected according to training dynamics."
  - Zhou, Tianyi and Wang, Shengjie and Bilmes, Jeff. *ICLR 2021*

- Robust Early-Learning: Hindering The Memorization of Noisy Labels.
  [[pdf]](https://openreview.net/pdf?id=Eql5b1_hTE4) [[online review]](https://openreview.net/forum?id=Eql5b1_hTE4)
  - "Robust early-learning: to reduce the side effect of noisy labels before early stopping and thus enhance the memorization of clean labels. Specifically, in each iteration, we divide all parameters into the critical and non-critical ones, and then perform different update rules for different types of parameters."
  - Xia, Xiaobo and Liu, Tongliang and Han, Bo and Gong, Chen and Wang, Nannan and Ge, Zongyuan and Chang, Yi. *ICLR 2021*

- When Do Curricula Work?
  [[pdf]](https://openreview.net/pdf?id=tW4QEInpni)
  - "We find that for standard benchmark datasets, curricula have only marginal benefits, and that randomly ordered samples perform as well or better than curricula and anti-curricula, suggesting that any benefit is entirely due to the dynamic training set size. ... Our experiments demonstrate that curriculum, but not anti-curriculum or random ordering can indeed improve the performance either with limited training time budget or in the existence of noisy data."
  - Wu, Xiaoxia and Dyer, Ethan and Neyshabur, Behnam. *ICLR 2021* (oral)

- Curriculum Graph Co-Teaching for Multi-Target Domain Adaptation.
  [[pdf]](https://arxiv.org/pdf/2104.00808.pdf) [[code]](https://github.com/Evgeneus/Graph-Domain-Adaptaion)
  - Roy, Subhankar and Krivosheev, Evgeny and Zhong, Zhun and Sebe, Nicu and Ricci, Elisa. *CVPR 2021*

<p align="center">
  <img src="https://github.com/Evgeneus/Graph-Domain-Adaptaion/blob/master/data/pipeline.png" alt="Curriculum Graph Co-Teaching" width="75%">
</p>

- TeachMyAgent: a Benchmark for Automatic Curriculum Learning in Deep RL.
  [[pdf]](https://arxiv.org/pdf/2103.09815.pdf) [[code]](https://github.com/flowersteam/TeachMyAgent)
  - Romac, Clément and Portelas, Rémy and Hofmann, Katja and Oudeyer, Pierre-Yves. *ICML 2021*

<p align="center">
  <img src="https://github.com/flowersteam/TeachMyAgent/blob/master/TeachMyAgent/graphics/readme_graphics/global_schema.png" alt="TechMyAgent" width="52%">
</p>

- Curriculum Learning by Optimizing Learning Dynamics.
  [[pdf]](http://proceedings.mlr.press/v130/zhou21a/zhou21a.pdf) [[code]](https://github.com/tianyizhou/DoCL)
  - Zhou, Tianyi and Wang, Shengjie and Bilmes, Jeff. *AISTATS 2021*

<p align="center">
  <img src="https://github.com/tianyizhou/DoCL/raw/main/docl_aistats2021_thumbnail.png" alt="DoCL" width="32%">
</p>
