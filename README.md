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

|  Tag  |        `Det`     |           `Seg`       |         `Cls`        |      `Trans`      |      `Gen`   |    `Other`    |
|:----------------:|:----------------:|:---------------------:|:--------------------:|:-----------------:|:------------:|:-----------:|
|  Item  | Detection | Semantic | Image Classification | Transfer Learning |  Generation  | other types |
|  Issues (e.g.,)  | long-tail | imbalance | imbalance, noise | long-tail, domain-shift |  collapose  |  -  |

### SURVEY
- Curriculum Learning: A Survey. *arxiv 2101.10382*
  [[pdf]](https://arxiv.org/pdf/2101.10382.pdf)

### 2009
- Curriculum Learning.
  [[pdf]](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/15972/Bengio%2C%202009%20Curriculum%20Learning.pdf?sequence=1&isAllowed=y) *ICML 2009*
  - "Rank the weights of the training examples, and use the rank to guide the order of presentation of examples to the learner."

### 2015
- Curriculum Learning.
  [[pdf]](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14374/14349) *AAAI 2015*

### 2017
- Self-Paced Learning: An Implicit Regularization Perspective.
  [[pdf]](https://www.researchgate.net/profile/Jian_Liang25/publication/303750070_Self-Paced_Learning_an_Implicit_Regularization_Perspective/links/5858e75b08ae3852d25555e3/Self-Paced-Learning-an-Implicit-Regularization-Perspective.pdf) *AAAI 2017*
  
- Curriculum Dropout.
  [[pdf]](http://www.vision.jhu.edu/assets/MorerioICCV17.pdf) [[code]](https://github.com/pmorerio/curriculum-dropout) *ICCV 2017*

- Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes.
  [[pdf]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Curriculum_Domain_Adaptation_ICCV_2017_paper.pdf) [[code]](https://github.com/YangZhang4065/AdaptationSeg) *ICCV 2017*

### 2018
- Curriculum Learning by Transfer Learning: Theory and Experiments with Deep Networks.
  [[pdf]](https://arxiv.org/pdf/1802.03796.pdf) *ICML 2018*
  - "Sort the training examples based on the *performance* of a pre-trained network on a larger dataset,
    and then finetune the model to the dataset at hand."

- Self-Paced Deep Learning for Weakly Supervised Object Detection.
  [[pdf]](https://arxiv.org/pdf/1605.07651.pdf) *TPAMI 2018*
  
- MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks.
  [[pdf]](http://proceedings.mlr.press/v80/jiang18c/jiang18c.pdf) [[code]](https://github.com/google/mentornet) *ICML 2018*
<p align="center">
  <img src="https://github.com/google/mentornet/blob/master/images/overview.png" alt="MentorNet" width="50%">
</p>

- CurriculumNet: Weakly Supervised Learning from Large-Scale Web Images.
  [[pdf]](https://arxiv.org/pdf/1808.01097.pdf) [[code]](https://github.com/MalongTech/research-curriculumnet) *ECCV 2018*

- Unsupervised Feature Selection by Self-Paced Learning Regularization.
  [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S0167865518302782) *Pattern Recognition Letters 2018*

- Learning to Teach with Dynamic Loss Functions.
  [[pdf]](https://papers.nips.cc/paper/7882-learning-to-teach-with-dynamic-loss-functions.pdf) *NeurIPS 2018*
  - "A good teacher not only provides his/her students with qualified teaching materials (e.g., textbooks), but also sets up appropriate learning objectives (e.g., course projects and exams) considering different situations of a student."

- Progressive Growing of GANs for Improved Quality, Stability, and Variation. `Gen`
  [[pdf]](https://openreview.net/forum?id=Hk99zCeAb&noteId=Hk99zCeAb) [[code]](https://github.com/tkarras/progressive_growing_of_gans) *ICLR 2018*
  - "The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses. This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented quality."
<p align="center">
  <img src="https://pic1.zhimg.com/80/v2-fdaeb2fb88c40b315420b89c96460105_1440w.jpg?source=1940ef5c" alt="Progressive growing of GANs" width="60%">
</p>

### 2019
- Transferable Curriculum for Weakly-Supervised Domain Adaptation.
  [[pdf]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/transferable-curriculum-aaai19.pdf) [[code]](https://github.com/thuml/TCL) *AAAI 2019*

- Dynamic Curriculum Learning for Imbalanced Data Classification.
  [[pdf]](https://arxiv.org/pdf/1901.06783.pdf)[[simple demo]](https://github.com/apeterswu/L2T_loss) *ICCV 2019*

- Guided Curriculum Model Adaptation and Uncertainty-Aware Evaluation for Semantic Nighttime Image Segmentation.
  [[pdf]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Sakaridis_Guided_Curriculum_Model_Adaptation_and_Uncertainty-Aware_Evaluation_for_Semantic_Nighttime_ICCV_2019_paper.pdf) *ICCV 2019*

- On The Power of Curriculum Learning in Training Deep Networks.
  [[pdf]](https://arxiv.org/pdf/1904.03626.pdf) *ICML 2019*

- Balanced Self-Paced Learning for Generative Adversarial Clustering Network.
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ghasedi_Balanced_Self-Paced_Learning_for_Generative_Adversarial_Clustering_Network_CVPR_2019_paper.pdf) *CVPR 2019*

### 2020
- Breaking the Curse of Space Explosion: Towards Effcient NAS with Curriculum Search.
  [[pdf]](http://proceedings.mlr.press/v119/guo20b.html) [[code]](https://github.com/guoyongcs/CNAS) *ICML 20*
<p align="center">
  <img src="https://github.com/guoyongcs/CNAS/blob/master/assets/cnas.jpg" alt="CNAS" width="45%">
</p>

- BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition.
  [[pdf]](https://arxiv.org/abs/1912.02413) [[code]](https://github.com/Megvii-Nanjing/BBN) *CVPR 2020*
<p align="center">
  <img src="https://github.com/Openning07/awesome-curriculum-learning/blob/master/images/BBN_CVPR20.png" alt="BBN" width="70%">
</p>

- Open Compound Domain Adaptation.
  [[pdf]](https://arxiv.org/abs/1909.03403) [[code]](https://github.com/zhmiao/OpenCompoundDomainAdaptation-OCDA) *CVPR 2020*
<p align="center">
  <img src="https://bair.berkeley.edu/static/blog/ocda/figure_4.png" alt="OCDA" width="65%">
</p>

- Curriculum Manager for Source Selection in Multi-Source Domain Adaptation.
  [[pdf]](https://arxiv.org/pdf/2007.01261v1.pdf)[[code]](https://github.com/LoyoYang/CMSS) *ECCV 2020*
  
- Content-Consistent Matching for Domain Adaptive Semantic Segmentation. `Seg`
  [[pdf]](https://arxiv.org/pdf/2007.01261v1.pdf) [[code]](https://github.com/Solacex/CCM) *ECCV 2020*
  - "to acquire those synthetic images that share similar distribution with the real ones in the target domain, so that the domain gap can be naturally alleviated by employing the content-consistent synthetic images for training."
  - "not all the source images could contribute to the improvement of adaptation performance, especially at certain training stages."
<p align="center">
  <img src="https://pic2.zhimg.com/80/v2-f6f3eb85a79f206b4f5524eaf43a71fd_1440w.jpg" alt="CMM" width="70%">
</p>

- DA-NAS: Data Adapted Pruning for Efficient Neural Architecture Search.
  [[pdf]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720579.pdf) *ECCV 2020*
  - "Our method is based on an interesting observation that the learning speed for blocks in deep neural networks is related to the difficulty of recognizing distinct categories. We carefully design a progressive data adapted pruning strategy for efficient architecture search. It will quickly trim low performed blocks on a subset of target dataset (e.g., easy classes), and then gradually find the best blocks on the whole target dataset."

- Label-similarity Curriculum Learning.
  [[pdf]](https://arxiv.org/pdf/1911.06902.pdf) [[code]](https://github.com/speedystream/LCL) *ECCV 2020*
  - "The idea is to use a probability distribution over classes as target label, where the class probabilities reflect the similarity to the true class. Gradually, this label
representation is shifted towards the standard one-hot-encoding."

- Semi-Supervised Semantic Segmentation via Dynamic Self-Training and Class-Balanced Curriculum.
  [[pdf]](https://arxiv.org/abs/2004.08514) [[code]](https://github.com/voldemortX/DST-CBC) *arXiv 2004.08514*
  
- Evolutionary Population Curriculum for Scaling Multi-Agent Reinforcement Learning.
  [[pdf]](https://arxiv.org/pdf/2003.10423.pdf)[[code]](https://github.com/qian18long/epciclr2020) *ICLR 2020*
  - "Evolutionary Population Curriculum (EPC), a curriculum learning paradigm that scales up MultiAgent Reinforcement Learning (MARL) by progressively increasing the population of training agents in a stage-wise manner."

- Automatic Curriculum Learning through Value Disagreement.
  [[pdf]](https://papers.nips.cc/paper/2020/file/566f0ea4f6c2e947f36795c8f58ba901-Paper.pdf) *NeurIPS 2020*
  - " When biological agents learn, there is often an organized and meaningful order to which learning happens."
  - "Our key insight is that if we can sample goals at the frontier of the set of goals that an agent is able to reach, it will provide a significantly stronger learning signal compared to randomly sampled goals"
<p align="center">
  <img src="https://github.com/Openning07/awesome-curriculum-learning/blob/master/images/AutomaticCurriculumLearningThroughValueDisagreement.png" alt="CMM" width="65%">
</p>

- Curriculum by Smoothing.
  [[pdf]](https://proceedings.neurips.cc/paper/2020/file/f6a673f09493afcd8b129a0bcf1cd5bc-Paper.pdf) [[code]](https://github.com/pairlab/CBS) *NeurIPS 2020*

- Curriculum Learning by Dynamic Instance Hardness.
  [[pdf]](https://papers.nips.cc/paper/2020/file/62000dee5a05a6a71de3a6127a68778a-Paper.pdf) *NeurIPS 2020*

- Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID.
  [[pdf]](https://arxiv.org/pdf/2006.02713.pdf) [[code]](https://github.com/yxgeee/SpCL) [[zhihu]](https://zhuanlan.zhihu.com/p/269112325?utm_source=wechat_session&utm_medium=social&utm_oi=41299705069568&utm_content=group3_article&utm_campaign=shareopn&wechatShare=2&s_r=0) *NeurIPS 2020*

- Self-Paced Deep Reinforcement Learning.
 [[pdf]](https://papers.nips.cc/paper/2020/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html) *NeurIPS 2020*

### 2021
- Robust Curriculum Learning: from clean label detection to noisy label self-correction.
  [[pdf]](https://openreview.net/pdf?id=lmTWnm3coJJ) [[online review]](https://openreview.net/forum?id=lmTWnm3coJJ) *ICLR 2021*
  - "Robust curriculum learning (RoCL) improves noisy label learning by periodical transitions from supervised learning of clean labeled data to self-supervision of wrongly-labeled data, where the data are selected according to training dynamics."

- Robust Early-Learning: Hindering The Memorization of Noisy Labels.
  [[pdf]](https://openreview.net/pdf?id=Eql5b1_hTE4) [[online review]](https://openreview.net/forum?id=Eql5b1_hTE4) *ICLR 2021*
  - "Robust early-learning: to reduce the side effect of noisy labels before early stopping and thus enhance the memorization of clean labels. Specifically, in each iteration, we divide all parameters into the critical and non-critical ones, and then perform different update rules for different types of parameters."

- When Do Curricula Work?
  [[pdf]](https://openreview.net/pdf?id=tW4QEInpni) *ICLR 2021* (oral)
  - "We find that for standard benchmark datasets, curricula have only marginal benefits, and that randomly ordered samples perform as well or better than curricula and anti-curricula, suggesting that any benefit is entirely due to the dynamic training set size. ... Our experiments demonstrate that curriculum, but not anti-curriculum or random ordering can indeed improve the performance either with limited training time budget or in the existence of noisy data."

- Curriculum Graph Co-Teaching for Multi-Target Domain Adaptation.
  [[pdf]](https://arxiv.org/pdf/2104.00808.pdf) [[code]](https://github.com/Evgeneus/Graph-Domain-Adaptaion) *CVPR 2021*
<p align="center">
  <img src="https://github.com/Evgeneus/Graph-Domain-Adaptaion/blob/master/data/pipeline.png" alt="Curriculum Graph Co-Teaching" width="75%">
</p>

- Unsupervised Curriculum Domain Adaptation for No-Reference Video Quality Assessment.
  [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Unsupervised_Curriculum_Domain_Adaptation_for_No-Reference_Video_Quality_Assessment_ICCV_2021_paper.pdf) [[code]](https://github.com/cpf0079/UCDA) *ICCV 2021*

- Adaptive Curriculum Learning.
  [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Kong_Adaptive_Curriculum_Learning_ICCV_2021_paper.pdf) *ICCV 2021*

- Multi-Level Curriculum for Training A Distortion-Aware Barrel Distortion Rectification Model.
  [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Liao_Multi-Level_Curriculum_for_Training_a_Distortion-Aware_Barrel_Distortion_Rectification_Model_ICCV_2021_paper.pdf) *ICCV 2021*

- TeachMyAgent: a Benchmark for Automatic Curriculum Learning in Deep RL.
  [[pdf]](https://arxiv.org/pdf/2103.09815.pdf) [[code]](https://github.com/flowersteam/TeachMyAgent) *ICML 2021*
<p align="center">
  <img src="https://github.com/flowersteam/TeachMyAgent/blob/master/TeachMyAgent/graphics/readme_graphics/global_schema.png" alt="TechMyAgent" width="52%">
</p>

- Self-Paced Context Evaluation for Contextual Reinforcement Learning.
  [[pdf]](https://arxiv.org/pdf/2106.05110.pdf) *ICML 2021*
  - "To improve sample efficiency for learning on such instances of a problem domain, we present Self-Paced Context Evaluation (SPaCE). Based on self-paced learning, \spc automatically generates \task curricula online with little computational overhead. To this end, SPaCE leverages information contained in state values during training to accelerate and improve training performance as well as generalization capabilities to new instances from the same problem domain."

- Curriculum Learning by Optimizing Learning Dynamics.
  [[pdf]](http://proceedings.mlr.press/v130/zhou21a/zhou21a.pdf) [[code]](https://github.com/tianyizhou/DoCL) *AISTATS 2021*
<p align="center">
  <img src="https://github.com/tianyizhou/DoCL/raw/main/docl_aistats2021_thumbnail.png" alt="DoCL" width="55%">
</p>

- FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling.
  [[pdf]](https://arxiv.org/pdf/2110.08263.pdf) [[code]](https://github.com/torchssl/torchssl) *NeurIPS 2021*
