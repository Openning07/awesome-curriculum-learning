# Awesome Curriculum Learning[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![](https://visitor-badge.glitch.me/badge?page_id=Openning07.awesome-curriculum-learning)
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
  [[pdf]](link) [[code]](link)
  - Key Contribution(s)
  - Author 1, Author 2, and Author 3. *Conference Year*
```

## Table of Contents

### Mainstreams of curriculum learning

|  Tag  |        `Det`     |           `Seg`       |         `Cls`        |      `Trans`      |      `Gen`   |   `RL`  |    `Other`    |
|:-----:|:----------------:|:---------------------:|:--------------------:|:-----------------:|:------------:|:-------:|:-------------:|
| Item  |    Detection     | Semantic | Classification | Transfer Learning |  Generation  | Reinforcement Learning | others |
|  Issues (e.g.,)  | long-tail | imbalance | imbalance, noise label | long-tail, domain-shift |  mode collapose  | exploit V.S. explore |  -  |

### SURVEY
- Curriculum Learning: A Survey. *arxiv 2101.10382*
  [[pdf]](https://arxiv.org/pdf/2101.10382.pdf)

### 2009
- Curriculum Learning.
  [[ICML]](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/15972/Bengio%2C%202009%20Curriculum%20Learning.pdf?sequence=1&isAllowed=y)
  - "Rank the weights of the training examples, and use the rank to guide the order of presentation of examples to the learner."

### 2011
- Learning the Easy Things First: Self-Paced Visual Category Discovery.
  [[CVPR]](https://vision.cs.utexas.edu/projects/easiness/easiness_cvpr2011.pdf)

### 2014
- Easy Samples First: Self-paced Reranking for Zero-Example Multimedia Search.
  [[ACM MM]](http://www.cs.cmu.edu/~lujiang/camera_ready_papers/ACM_MM_fp_2014.pdf)

### 2015
- Self-paced Curriculum Learning.
  [[AAAI]](http://www.cs.cmu.edu/~lujiang/camera_ready_papers/AAAI_SPCL_2015.pdf)

- Curriculum Learning of Multiple Tasks.
  [[CVPR]](https://openaccess.thecvf.com/content_cvpr_2015/papers/Pentina_Curriculum_Learning_of_2015_CVPR_paper.pdf)

- A Self-paced Multiple-instance Learning Framework for Co-saliency Detection.
  [[ICCV]](https://openaccess.thecvf.com/content_iccv_2015/papers/Zhang_A_Self-Paced_Multiple-Instance_ICCV_2015_paper.pdf)

### 2016
- Multi-modal Curriculum Learning for Semi-supervised Image Classification.
  [[TIP]](https://www.dcs.bbk.ac.uk/~sjmaybank/MultiModal.pdf)

### 2017
- Self-Paced Learning: An Implicit Regularization Perspective.
  [[AAAI]](https://www.researchgate.net/profile/Jian_Liang25/publication/303750070_Self-Paced_Learning_an_Implicit_Regularization_Perspective/links/5858e75b08ae3852d25555e3/Self-Paced-Learning-an-Implicit-Regularization-Perspective.pdf)
  
- SPFTN: A Self-Paced Fine-Tuning Network for Segmenting Objects in Weakly Labelled Videos.
  [[CVPR]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_SPFTN_A_Self-Paced_CVPR_2017_paper.pdf) [[code]](https://github.com/VividLe/SPFTN)
  
- Curriculum Dropout.
  [[ICCV]](http://www.vision.jhu.edu/assets/MorerioICCV17.pdf) [[code]](https://github.com/pmorerio/curriculum-dropout)

- Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes.
  [[ICCV]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Curriculum_Domain_Adaptation_ICCV_2017_paper.pdf) [[code]](https://github.com/YangZhang4065/AdaptationSeg)

- Self-Paced Co-training.
  [[ICML]](http://proceedings.mlr.press/v70/ma17b/ma17b.pdf) [[code]](https://github.com/Flowerfan/Open-Reid)

- Active Self-Paced Learning for Cost-Effective and Progressive Face Identification.
  [[TPAMI]](https://arxiv.org/pdf/1701.03555.pdf) [[code]](https://github.com/kezewang/ASPL)

- Co-saliency detection via a self-paced multiple-instance learning framework.
  [[TPAMI]](https://ieeexplore.ieee.org/abstract/document/7469327)

- A Self-Paced Regularization Framework for Multi-Label Learning.
  [[TNNLS]](https://arxiv.org/pdf/1603.06708.pdf)

- Reverse Curriculum Generation for Reinforcement Learning.
  [[CoRL]](https://arxiv.org/pdf/1707.05300.pdf)

### 2018
- Curriculum Learning by Transfer Learning: Theory and Experiments with Deep Networks.
  [[ICML]](https://arxiv.org/pdf/1802.03796.pdf)
  - "Sort the training examples based on the *performance* of a pre-trained network on a larger dataset,
    and then finetune the model to the dataset at hand."
  
- MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks.
  [[ICML]](http://proceedings.mlr.press/v80/jiang18c/jiang18c.pdf) [[code]](https://github.com/google/mentornet)
<p align="center">
  <img src="https://github.com/google/mentornet/blob/master/images/overview.png" alt="MentorNet" width="50%">
</p>

- CurriculumNet: Weakly Supervised Learning from Large-Scale Web Images.
  [[ECCV]](https://arxiv.org/pdf/1808.01097.pdf) [[code]](https://github.com/MalongTech/research-curriculumnet)

- Progressive Growing of GANs for Improved Quality, Stability, and Variation. `Gen`
  [[ICLR]](https://openreview.net/forum?id=Hk99zCeAb&noteId=Hk99zCeAb) [[code]](https://github.com/tkarras/progressive_growing_of_gans)
  - "The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses. This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented quality."
<p align="center">
  <img src="https://pic1.zhimg.com/80/v2-fdaeb2fb88c40b315420b89c96460105_1440w.jpg?source=1940ef5c" alt="Progressive growing of GANs" width="60%">
</p>

- Minimax curriculum learning: Machine teaching with desirable difficulties and scheduled diversity.
  [[ICLR]](https://openreview.net/pdf?id=BywyFQlAW)

- Learning to Teach with Dynamic Loss Functions.
  [[NeurIPS]](https://papers.nips.cc/paper/7882-learning-to-teach-with-dynamic-loss-functions.pdf)
  - "A good teacher not only provides his/her students with qualified teaching materials (e.g., textbooks), but also sets up appropriate learning objectives (e.g., course projects and exams) considering different situations of a student."

- Self-Paced Deep Learning for Weakly Supervised Object Detection.
  [[TPAMI]](https://arxiv.org/pdf/1605.07651.pdf)

- Unsupervised Feature Selection by Self-Paced Learning Regularization.
  [[Pattern Recognition Letters]](https://www.sciencedirect.com/science/article/abs/pii/S0167865518302782)

### 2019
- Transferable Curriculum for Weakly-Supervised Domain Adaptation.
  [[AAAI]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/transferable-curriculum-aaai19.pdf) [[code]](https://github.com/thuml/TCL)

- Balanced Self-Paced Learning for Generative Adversarial Clustering Network.
  [[CVPR]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ghasedi_Balanced_Self-Paced_Learning_for_Generative_Adversarial_Clustering_Network_CVPR_2019_paper.pdf)

- Local to Global Learning: Gradually Adding Classes for Training Deep Neural Networks.
  [[CVPR]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cheng_Local_to_Global_Learning_Gradually_Adding_Classes_for_Training_Deep_CVPR_2019_paper.pdf) [[code]](https://github.com/piratehao/Local-to-Global-Learning-for-DNNs)

- Dynamic Curriculum Learning for Imbalanced Data Classification.
  [[ICCV]](https://arxiv.org/pdf/1901.06783.pdf) [[simple demo]](https://github.com/apeterswu/L2T_loss)

- Guided Curriculum Model Adaptation and Uncertainty-Aware Evaluation for Semantic Nighttime Image Segmentation.
  [[ICCV]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Sakaridis_Guided_Curriculum_Model_Adaptation_and_Uncertainty-Aware_Evaluation_for_Semantic_Nighttime_ICCV_2019_paper.pdf) [[code]](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/)

- On The Power of Curriculum Learning in Training Deep Networks.
  [[pdf]](https://arxiv.org/pdf/1904.03626.pdf) *ICML*

- Data Parameters: A New Family of Parameters for Learning a Differentiable Curriculum.
  [[NeurIPS]](https://papers.nips.cc/paper/2019/file/926ffc0ca56636b9e73c565cf994ea5a-Paper.pdf) [[code]](https://github.com/apple/ml-data-parameters)

-Leveraging prior-knowledge for weakly supervised object detection under a collaborative self-paced curriculum learning framework.
  [[IJCV]](https://openreview.net/forum?id=Jv2tq4Opli)

- Curriculum Model Adaptation with Synthetic and Real Data for Semantic Foggy Scene Understanding.
  [[IJCV]](https://arxiv.org/pdf/1901.01415.pdf)

### 2020
- Breaking the Curse of Space Explosion: Towards Effcient NAS with Curriculum Search.
  [[ICML]](http://proceedings.mlr.press/v119/guo20b.html) [[code]](https://github.com/guoyongcs/CNAS)
<p align="center">
  <img src="https://github.com/guoyongcs/CNAS/blob/master/assets/cnas.jpg" alt="CNAS" width="45%">
</p>

- BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition.
  [[CVPR]](https://arxiv.org/abs/1912.02413) [[code]](https://github.com/Megvii-Nanjing/BBN)
<p align="center">
  <img src="https://github.com/Openning07/awesome-curriculum-learning/blob/master/images/BBN_CVPR20.png" alt="BBN" width="70%">
</p>

- Open Compound Domain Adaptation.
  [[CVPR]](https://arxiv.org/abs/1909.03403) [[code]](https://github.com/zhmiao/OpenCompoundDomainAdaptation-OCDA)
<p align="center">
  <img src="https://bair.berkeley.edu/static/blog/ocda/figure_4.png" alt="OCDA" width="65%">
</p>

- Curricularface: adaptive curriculum learning loss for deep face recognition.
  [[CVPR]](https://arxiv.org/pdf/2004.00288.pdf) [[code]](https://github.com/HuangYG123/CurricularFace)
  - "our CurricularFace adaptively adjusts the relative importance of easy and hard samples during different training stages. In each stage, different samples are assigned with different importance according to their corresponding difficultness."

- Curriculum Manager for Source Selection in Multi-Source Domain Adaptation.
  [[ECCV]](https://arxiv.org/pdf/2007.01261v1.pdf)[[code]](https://github.com/LoyoYang/CMSS)
  
- Content-Consistent Matching for Domain Adaptive Semantic Segmentation. `Seg`
  [[ECCV]](https://arxiv.org/pdf/2007.01261v1.pdf) [[code]](https://github.com/Solacex/CCM)
  - "to acquire those synthetic images that share similar distribution with the real ones in the target domain, so that the domain gap can be naturally alleviated by employing the content-consistent synthetic images for training."
  - "not all the source images could contribute to the improvement of adaptation performance, especially at certain training stages."
<p align="center">
  <img src="https://pic2.zhimg.com/80/v2-f6f3eb85a79f206b4f5524eaf43a71fd_1440w.jpg" alt="CMM" width="70%">
</p>

- DA-NAS: Data Adapted Pruning for Efficient Neural Architecture Search.
  [[ECCV]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720579.pdf)
  - "Our method is based on an interesting observation that the learning speed for blocks in deep neural networks is related to the difficulty of recognizing distinct categories. We carefully design a progressive data adapted pruning strategy for efficient architecture search. It will quickly trim low performed blocks on a subset of target dataset (e.g., easy classes), and then gradually find the best blocks on the whole target dataset."

- Label-similarity Curriculum Learning.
  [[ECCV]](https://arxiv.org/pdf/1911.06902.pdf) [[code]](https://github.com/speedystream/LCL)
  - "The idea is to use a probability distribution over classes as target label, where the class probabilities reflect the similarity to the true class. Gradually, this label representation is shifted towards the standard one-hot-encoding."

- Multi-Task Curriculum Framework for Open-Set Semi-Supervised Learning.
  [[ECCV]](https://arxiv.org/pdf/2007.11330.pdf) [[code]](https://github.com/YU1ut/Multi-Task-Curriculum-Framework-for-Open-Set-SSL)

- Semi-Supervised Semantic Segmentation via Dynamic Self-Training and Class-Balanced Curriculum.
  [[arXiv]](https://arxiv.org/abs/2004.08514) [[code]](https://github.com/voldemortX/DST-CBC)
  
- Evolutionary Population Curriculum for Scaling Multi-Agent Reinforcement Learning.
  [[ICLR]](https://arxiv.org/pdf/2003.10423.pdf)[[code]](https://github.com/qian18long/epciclr2020)
  - "Evolutionary Population Curriculum (EPC), a curriculum learning paradigm that scales up MultiAgent Reinforcement Learning (MARL) by progressively increasing the population of training agents in a stage-wise manner."

- Curriculum Loss: Robust Learning and Generalization Against Label Corruption.
  [[ICLR]](https://arxiv.org/pdf/1905.10045.pdf)

- Automatic Curriculum Learning through Value Disagreement.
  [[NeurIPS]](https://papers.nips.cc/paper/2020/file/566f0ea4f6c2e947f36795c8f58ba901-Paper.pdf)
  - " When biological agents learn, there is often an organized and meaningful order to which learning happens."
  - "Our key insight is that if we can sample goals at the frontier of the set of goals that an agent is able to reach, it will provide a significantly stronger learning signal compared to randomly sampled goals"
<p align="center">
  <img src="https://github.com/Openning07/awesome-curriculum-learning/blob/master/images/AutomaticCurriculumLearningThroughValueDisagreement.png" alt="CMM" width="65%">
</p>

- Curriculum by Smoothing.
  [[NeurIPS]](https://proceedings.neurips.cc/paper/2020/file/f6a673f09493afcd8b129a0bcf1cd5bc-Paper.pdf) [[code]](https://github.com/pairlab/CBS)

- Curriculum Learning by Dynamic Instance Hardness.
  [[NeurIPS]](https://papers.nips.cc/paper/2020/file/62000dee5a05a6a71de3a6127a68778a-Paper.pdf)

- Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID.
  [[NeurIPS]](https://arxiv.org/pdf/2006.02713.pdf) [[code]](https://github.com/yxgeee/SpCL) [[zhihu]](https://zhuanlan.zhihu.com/p/269112325?utm_source=wechat_session&utm_medium=social&utm_oi=41299705069568&utm_content=group3_article&utm_campaign=shareopn&wechatShare=2&s_r=0)

- Self-Paced Deep Reinforcement Learning.
 [[NeurIPS]](https://papers.nips.cc/paper/2020/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html)
 
- SuperLoss: A Generic Loss for Robust Curriculum Learning.
  [[NeurIPS]](https://proceedings.neurips.cc/paper/2020/file/2cfa8f9e50e0f510ede9d12338a5f564-Paper.pdf) [[code]](https://github.com/AlanChou/Super-Loss)

- Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey.
  [[JMLR]](https://jmlr.org/papers/volume21/20-212/20-212.pdf)

### 2021
- Curriculum Labeling: Revisiting Pseudo-Labeling for Semi-Supervised Learning.
  [[AAAI]](https://arxiv.org/pdf/2001.06001.pdf) [[code]](https://github.com/uvavision/Curriculum-Labeling)

- Robust Curriculum Learning: from clean label detection to noisy label self-correction.
  [[ICLR]](https://openreview.net/pdf?id=lmTWnm3coJJ) [[online review]](https://openreview.net/forum?id=lmTWnm3coJJ)
  - "Robust curriculum learning (RoCL) improves noisy label learning by periodical transitions from supervised learning of clean labeled data to self-supervision of wrongly-labeled data, where the data are selected according to training dynamics."

- Robust Early-Learning: Hindering The Memorization of Noisy Labels.
  [[ICLR]](https://openreview.net/pdf?id=Eql5b1_hTE4) [[online review]](https://openreview.net/forum?id=Eql5b1_hTE4)
  - "Robust early-learning: to reduce the side effect of noisy labels before early stopping and thus enhance the memorization of clean labels. Specifically, in each iteration, we divide all parameters into the critical and non-critical ones, and then perform different update rules for different types of parameters."

- When Do Curricula Work?
  [[ICLR (oral)]](https://openreview.net/pdf?id=tW4QEInpni)
  - "We find that for standard benchmark datasets, curricula have only marginal benefits, and that randomly ordered samples perform as well or better than curricula and anti-curricula, suggesting that any benefit is entirely due to the dynamic training set size. ... Our experiments demonstrate that curriculum, but not anti-curriculum or random ordering can indeed improve the performance either with limited training time budget or in the existence of noisy data."

- Curriculum Graph Co-Teaching for Multi-Target Domain Adaptation. `TL`
  [[CVPR]](https://arxiv.org/pdf/2104.00808.pdf) [[code]](https://github.com/Evgeneus/Graph-Domain-Adaptaion)
<p align="center">
  <img src="https://github.com/Evgeneus/Graph-Domain-Adaptaion/blob/master/data/pipeline.png" alt="Curriculum Graph Co-Teaching" width="80%">
</p>

- Unsupervised Curriculum Domain Adaptation for No-Reference Video Quality Assessment. `Cls`
  [[ICCV]](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Unsupervised_Curriculum_Domain_Adaptation_for_No-Reference_Video_Quality_Assessment_ICCV_2021_paper.pdf) [[code]](https://github.com/cpf0079/UCDA)

- Adaptive Curriculum Learning.
  [[ICCV]](https://openaccess.thecvf.com/content/ICCV2021/papers/Kong_Adaptive_Curriculum_Learning_ICCV_2021_paper.pdf)

- Multi-Level Curriculum for Training A Distortion-Aware Barrel Distortion Rectification Model.
  [[ICCV]](https://openaccess.thecvf.com/content/ICCV2021/papers/Liao_Multi-Level_Curriculum_for_Training_a_Distortion-Aware_Barrel_Distortion_Rectification_Model_ICCV_2021_paper.pdf)

- TeachMyAgent: a Benchmark for Automatic Curriculum Learning in Deep RL.
  [[ICML]](https://arxiv.org/pdf/2103.09815.pdf) [[code]](https://github.com/flowersteam/TeachMyAgent)
<p align="center">
  <img src="https://github.com/flowersteam/TeachMyAgent/blob/master/TeachMyAgent/graphics/readme_graphics/global_schema.png" alt="TechMyAgent" width="52%">
</p>

- Self-Paced Context Evaluation for Contextual Reinforcement Learning.  `RL`
  [[ICCV]](https://arxiv.org/pdf/2106.05110.pdf)
  - "To improve sample efficiency for learning on such instances of a problem domain, we present Self-Paced Context Evaluation (SPaCE). Based on self-paced learning, \spc automatically generates \task curricula online with little computational overhead. To this end, SPaCE leverages information contained in state values during training to accelerate and improve training performance as well as generalization capabilities to new instances from the same problem domain."

- Curriculum Learning by Optimizing Learning Dynamics.
  [[AISTATS]](http://proceedings.mlr.press/v130/zhou21a/zhou21a.pdf) [[code]](https://github.com/tianyizhou/DoCL)
<p align="center">
  <img src="https://github.com/tianyizhou/DoCL/raw/main/docl_aistats2021_thumbnail.png" alt="DoCL" width="52%">
</p>

- FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling. `Cls`
  [[NeurIPS]](https://arxiv.org/pdf/2110.08263.pdf) [[code]](https://github.com/torchssl/torchssl)

- Learning with Noisy Correspondence for Cross-modal Matching.
  [[NeurIPS]](https://proceedings.neurips.cc/paper/2021/file/f5e62af885293cf4d511ceef31e61c80-Paper.pdf) [[code]](https://github.com/XLearning-SCU/2021-NeurIPS-NCR)
<p align="center">
  <img src="https://github.com/Openning07/awesome-curriculum-learning/blob/master/images/NCR_NeurIPS21.png" alt="NCR" width="85%">
</p>

- Self-Paced Contrastive Learning for Semi-Supervised Medical Image Segmentation with Meta-labels.
  [[NeurIPS]](https://proceedings.neurips.cc/paper/2021/file/8b5c8441a8ff8e151b191c53c1842a38-Paper.pdf)
  - "A self-paced learning strategy exploiting the weak annotations is proposed to further help the learning process and discriminate useful labels from noise."

- Curriculum Learning for Vision-and-Language Navigation.
  [[NeurIPS]](https://proceedings.neurips.cc/paper/2021/file/6f0442558302a6ededff195daf67f79b-Paper.pdf)
  - "We propose a novel curriculum- based training paradigm for VLN tasks that can balance human prior knowledge and agent learning progress about training samples."

### 2022
- Characterizing and overcoming the greedy nature of learning in multi-modal deep neural networks.
  [[ICML]](https://arxiv.org/pdf/2202.05306.pdf) [[code]](https://github.com/nyukat/greedy_multimodal_learning)
  - "We hypothesize that due to the greedy nature of learning in multi-modal deep neural networks, these models tend to rely on just one modality while under-fitting the other modalities. ... We propose an algorithm to balance the conditional learning speeds between modalities during training and demonstrate that it indeed addresses the issue of greedy learning."

- Pseudo-Labeled Auto-Curriculum Learning for Semi-Supervised Keypoint Localization.  `Seg`
  [[ICLR]](https://arxiv.org/pdf/2201.08613.pdf) [[open review]](https://openreview.net/forum?id=6Q52pZ-Th7N)
  - "We propose to automatically select reliable pseudo-labeled samples with a series of dynamic thresholds, which constitutes a learning curriculum."

- C-Planning: An Automatic Curriculum for Learning Goal-Reaching Tasks. `RL`
  [[ICLR]](https://openreview.net/pdf?id=K2JfSnLBD9) [[open review]](https://openreview.net/forum?id=K2JfSnLBD9)

- Curriculum learning as a tool to uncover learning principles in the brain.
  [[ICLR]](https://openreview.net/pdf?id=TpJMvo0_pu-) [[open review]](https://openreview.net/forum?id=TpJMvo0_pu-)
  
- It Takes Four to Tango: Multiagent Self Play for Automatic Curriculum Generation.
  [[ICLR]](https://openreview.net/pdf?id=q4tZR1Y-UIs) [[open review]](https://openreview.net/forum?id=q4tZR1Y-UIs)
  
- Boosted Curriculum Reinforcement Learning. `RL`
  [[ICLR]](https://openreview.net/pdf?id=anbBFlX1tJ1) [[open review]](https://openreview.net/forum?id=anbBFlX1tJ1)

- ST++: Make Self-Training Work Better for Semi-Supervised Semantic Segmentation.
  [[CVPR]](https://arxiv.org/pdf/2106.05095.pdf) [[code]](https://github.com/LiheYoung/ST-PlusPlus)
  - "propose a selective re-training scheme via prioritizing reliable unlabeled samples to safely exploit the whole unlabeled set in an easy-to-hard curriculum learning manner."

- Robust Cross-Modal Representation Learning with Progressive Self-Distillation.
  [[CVPR]](https://openaccess.thecvf.com/content/CVPR2022/papers/Andonian_Robust_Cross-Modal_Representation_Learning_With_Progressive_Self-Distillation_CVPR_2022_paper.pdf)
  - "The learning objective of vision-language approach of CLIP does not effectively account for the noisy many-to-many correspondences found in web-harvested image captioning datasets. To address this challenge, we introduce a novel training framework based on cross-modal contrastive learning that uses progressive self-distillation and soft image-text alignments to more efficiently learn robust representations from noisy data."

- EAT-C: Environment-Adversarial sub-Task Curriculum for Efficient Reinforcement Learning.
  [[ICML]](https://proceedings.mlr.press/v162/ao22a/ao22a.pdf)
  
- Curriculum Reinforcement Learning via Constrained Optimal Transport.
  [[ICML]](https://proceedings.mlr.press/v162/klink22a/klink22a.pdf) [[code]](https://github.com/psclklnk/currot)
  
- Evolving Curricula with Regret-Based Environment Design.
  [[ICML]](https://proceedings.mlr.press/v162/parker-holder22a/parker-holder22a.pdf) [[project]](https://accelagent.github.io/)
  
- Robust Deep Reinforcement Learning through Bootstrapped Opportunistic Curriculum.
  [[ICML]](https://proceedings.mlr.press/v162/wu22k/wu22k.pdf) [[code]](https://github.com/jlwu002/BCL)

- On the Statistical Benefits of Curriculum Learning.
  [[ICML]](https://proceedings.mlr.press/v162/xu22i/xu22i.pdf)

- CLOSE: Curriculum Learning On the Sharing Extent Towards Better One-shot NAS.
  [[ECCV]](https://arxiv.org/pdf/2207.07868.pdf) [[code]](https://github.com/walkerning/aw_nas)
  - "We propose to apply Curriculum Learning On Sharing Extent (CLOSE) to train the supernet both efficiently and effectively. Specifically, we train the supernet with a large sharing extent (an easier curriculum) at the beginning and gradually decrease the sharing extent of the supernet (a harder curriculum)."

- Curriculum Learning for Data-Efficient Vision-Language Alignment.
  [[arxiv]](https://arxiv.org/pdf/2207.14525.pdf)
