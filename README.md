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
  [[pdf]](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/15972/Bengio%2C%202009%20Curriculum%20Learning.pdf?sequence=1&isAllowed=y) *ICML*
  - "Rank the weights of the training examples, and use the rank to guide the order of presentation of examples to the learner."

### 2011
- Learning the Easy Things First: Self-Paced Visual Category Discovery.
  [[pdf]](https://vision.cs.utexas.edu/projects/easiness/easiness_cvpr2011.pdf) *CVPR*

### 2014
- Easy Samples First: Self-paced Reranking for Zero-Example Multimedia Search.
  [[pdf]](http://www.cs.cmu.edu/~lujiang/camera_ready_papers/ACM_MM_fp_2014.pdf) *ACM MM*

### 2015
- Self-paced Curriculum Learning.
  [[pdf]](http://www.cs.cmu.edu/~lujiang/camera_ready_papers/AAAI_SPCL_2015.pdf) *AAAI*

- Curriculum Learning of Multiple Tasks.
  [[pdf]](https://openaccess.thecvf.com/content_cvpr_2015/papers/Pentina_Curriculum_Learning_of_2015_CVPR_paper.pdf) *CVPR*

- A Self-paced Multiple-instance Learning Framework for Co-saliency Detection.
  [[pdf]](https://openaccess.thecvf.com/content_iccv_2015/papers/Zhang_A_Self-Paced_Multiple-Instance_ICCV_2015_paper.pdf) *ICCV*

### 2016
- Multi-modal Curriculum Learning for Semi-supervised Image Classification.
  [[pdf]](https://www.dcs.bbk.ac.uk/~sjmaybank/MultiModal.pdf) *TIP*

### 2017
- Self-Paced Learning: An Implicit Regularization Perspective.
  [[pdf]](https://www.researchgate.net/profile/Jian_Liang25/publication/303750070_Self-Paced_Learning_an_Implicit_Regularization_Perspective/links/5858e75b08ae3852d25555e3/Self-Paced-Learning-an-Implicit-Regularization-Perspective.pdf) *AAAI*
  
- SPFTN: A Self-Paced Fine-Tuning Network for Segmenting Objects in Weakly Labelled Videos.
  [[pdf]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_SPFTN_A_Self-Paced_CVPR_2017_paper.pdf) [[code]](https://github.com/VividLe/SPFTN) *CVPR*
  
- Curriculum Dropout.
  [[pdf]](http://www.vision.jhu.edu/assets/MorerioICCV17.pdf) [[code]](https://github.com/pmorerio/curriculum-dropout) *ICCV*

- Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes.
  [[pdf]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Curriculum_Domain_Adaptation_ICCV_2017_paper.pdf) [[code]](https://github.com/YangZhang4065/AdaptationSeg) *ICCV*

- Self-Paced Co-training.
  [[pdf]](http://proceedings.mlr.press/v70/ma17b/ma17b.pdf) [[code]](https://github.com/Flowerfan/Open-Reid) *ICML*

- Active Self-Paced Learning for Cost-Effective and Progressive Face Identification.
  [[pdf]](https://arxiv.org/pdf/1701.03555.pdf) [[code]](https://github.com/kezewang/ASPL) *TPAMI*

- Co-saliency detection via a self-paced multiple-instance learning framework.
  [[link]](https://ieeexplore.ieee.org/abstract/document/7469327) *TPAMI*

- A Self-Paced Regularization Framework for Multi-Label Learning.
  [[pdf]](https://arxiv.org/pdf/1603.06708.pdf) *TNNLS*

- Reverse Curriculum Generation for Reinforcement Learning.
  [[https://arxiv.org/pdf/1707.05300.pdf]] *CoRL*

### 2018
- Curriculum Learning by Transfer Learning: Theory and Experiments with Deep Networks.
  [[pdf]](https://arxiv.org/pdf/1802.03796.pdf) *ICML*
  - "Sort the training examples based on the *performance* of a pre-trained network on a larger dataset,
    and then finetune the model to the dataset at hand."
  
- MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks.
  [[pdf]](http://proceedings.mlr.press/v80/jiang18c/jiang18c.pdf) [[code]](https://github.com/google/mentornet) *ICML*
<p align="center">
  <img src="https://github.com/google/mentornet/blob/master/images/overview.png" alt="MentorNet" width="50%">
</p>

- CurriculumNet: Weakly Supervised Learning from Large-Scale Web Images.
  [[pdf]](https://arxiv.org/pdf/1808.01097.pdf) [[code]](https://github.com/MalongTech/research-curriculumnet) *ECCV*

- Progressive Growing of GANs for Improved Quality, Stability, and Variation. `Gen`
  [[pdf]](https://openreview.net/forum?id=Hk99zCeAb&noteId=Hk99zCeAb) [[code]](https://github.com/tkarras/progressive_growing_of_gans) *ICLR*
  - "The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses. This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented quality."
<p align="center">
  <img src="https://pic1.zhimg.com/80/v2-fdaeb2fb88c40b315420b89c96460105_1440w.jpg?source=1940ef5c" alt="Progressive growing of GANs" width="60%">
</p>

- Minimax curriculum learning: Machine teaching with desirable difficulties and scheduled diversity.
  [[pdf]](https://openreview.net/pdf?id=BywyFQlAW) *ICLR*

- Learning to Teach with Dynamic Loss Functions.
  [[pdf]](https://papers.nips.cc/paper/7882-learning-to-teach-with-dynamic-loss-functions.pdf) *NeurIPS*
  - "A good teacher not only provides his/her students with qualified teaching materials (e.g., textbooks), but also sets up appropriate learning objectives (e.g., course projects and exams) considering different situations of a student."

- Self-Paced Deep Learning for Weakly Supervised Object Detection.
  [[pdf]](https://arxiv.org/pdf/1605.07651.pdf) *TPAMI*

- Unsupervised Feature Selection by Self-Paced Learning Regularization.
  [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S0167865518302782) *Pattern Recognition Letters*

### 2019
- Transferable Curriculum for Weakly-Supervised Domain Adaptation.
  [[pdf]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/transferable-curriculum-aaai19.pdf) [[code]](https://github.com/thuml/TCL) *AAAI*

- Balanced Self-Paced Learning for Generative Adversarial Clustering Network.
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ghasedi_Balanced_Self-Paced_Learning_for_Generative_Adversarial_Clustering_Network_CVPR_2019_paper.pdf) *CVPR*

- Local to Global Learning: Gradually Adding Classes for Training Deep Neural Networks.
  [[pdf]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cheng_Local_to_Global_Learning_Gradually_Adding_Classes_for_Training_Deep_CVPR_2019_paper.pdf) [[code]](https://github.com/piratehao/Local-to-Global-Learning-for-DNNs) *CVPR*

- Dynamic Curriculum Learning for Imbalanced Data Classification.
  [[pdf]](https://arxiv.org/pdf/1901.06783.pdf) [[simple demo]](https://github.com/apeterswu/L2T_loss) *ICCV*

- Guided Curriculum Model Adaptation and Uncertainty-Aware Evaluation for Semantic Nighttime Image Segmentation.
  [[pdf]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Sakaridis_Guided_Curriculum_Model_Adaptation_and_Uncertainty-Aware_Evaluation_for_Semantic_Nighttime_ICCV_2019_paper.pdf) [[code]](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/) *ICCV*

- On The Power of Curriculum Learning in Training Deep Networks.
  [[pdf]](https://arxiv.org/pdf/1904.03626.pdf) *ICML*

- Data Parameters: A New Family of Parameters for Learning a Differentiable Curriculum.
  [[pdf]](https://papers.nips.cc/paper/2019/file/926ffc0ca56636b9e73c565cf994ea5a-Paper.pdf) [[code]](https://github.com/apple/ml-data-parameters) *NeurIPS*

-Leveraging prior-knowledge for weakly supervised object detection under a collaborative self-paced curriculum learning framework.
  [[link]](https://openreview.net/forum?id=Jv2tq4Opli) *IJCV*

- Curriculum Model Adaptation with Synthetic and Real Data for Semantic Foggy Scene Understanding.
  [[pdf]](https://arxiv.org/pdf/1901.01415.pdf) *IJCV*

### 2020
- Breaking the Curse of Space Explosion: Towards Effcient NAS with Curriculum Search.
  [[pdf]](http://proceedings.mlr.press/v119/guo20b.html) [[code]](https://github.com/guoyongcs/CNAS) *ICML*
<p align="center">
  <img src="https://github.com/guoyongcs/CNAS/blob/master/assets/cnas.jpg" alt="CNAS" width="45%">
</p>

- BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition.
  [[pdf]](https://arxiv.org/abs/1912.02413) [[code]](https://github.com/Megvii-Nanjing/BBN) *CVPR*
<p align="center">
  <img src="https://github.com/Openning07/awesome-curriculum-learning/blob/master/images/BBN_CVPR20.png" alt="BBN" width="70%">
</p>

- Open Compound Domain Adaptation.
  [[pdf]](https://arxiv.org/abs/1909.03403) [[code]](https://github.com/zhmiao/OpenCompoundDomainAdaptation-OCDA) *CVPR*
<p align="center">
  <img src="https://bair.berkeley.edu/static/blog/ocda/figure_4.png" alt="OCDA" width="65%">
</p>

- Curricularface: adaptive curriculum learning loss for deep face recognition.
  [[pdf]](https://arxiv.org/pdf/2004.00288.pdf) [[code]](https://github.com/HuangYG123/CurricularFace) *CVPR*
  - "our CurricularFace adaptively adjusts the relative importance of easy and hard samples during different training stages. In each stage, different samples are assigned with different importance according to their corresponding difficultness."

- Curriculum Manager for Source Selection in Multi-Source Domain Adaptation.
  [[pdf]](https://arxiv.org/pdf/2007.01261v1.pdf)[[code]](https://github.com/LoyoYang/CMSS) *ECCV*
  
- Content-Consistent Matching for Domain Adaptive Semantic Segmentation. `Seg`
  [[pdf]](https://arxiv.org/pdf/2007.01261v1.pdf) [[code]](https://github.com/Solacex/CCM) *ECCV*
  - "to acquire those synthetic images that share similar distribution with the real ones in the target domain, so that the domain gap can be naturally alleviated by employing the content-consistent synthetic images for training."
  - "not all the source images could contribute to the improvement of adaptation performance, especially at certain training stages."
<p align="center">
  <img src="https://pic2.zhimg.com/80/v2-f6f3eb85a79f206b4f5524eaf43a71fd_1440w.jpg" alt="CMM" width="70%">
</p>

- DA-NAS: Data Adapted Pruning for Efficient Neural Architecture Search.
  [[pdf]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720579.pdf) *ECCV*
  - "Our method is based on an interesting observation that the learning speed for blocks in deep neural networks is related to the difficulty of recognizing distinct categories. We carefully design a progressive data adapted pruning strategy for efficient architecture search. It will quickly trim low performed blocks on a subset of target dataset (e.g., easy classes), and then gradually find the best blocks on the whole target dataset."

- Label-similarity Curriculum Learning.
  [[pdf]](https://arxiv.org/pdf/1911.06902.pdf) [[code]](https://github.com/speedystream/LCL) *ECCV*
  - "The idea is to use a probability distribution over classes as target label, where the class probabilities reflect the similarity to the true class. Gradually, this label
representation is shifted towards the standard one-hot-encoding."

- Multi-Task Curriculum Framework for Open-Set Semi-Supervised Learning.
  [[pdf]](https://arxiv.org/pdf/2007.11330.pdf) [[code]](https://github.com/YU1ut/Multi-Task-Curriculum-Framework-for-Open-Set-SSL) *ECCV*

- Semi-Supervised Semantic Segmentation via Dynamic Self-Training and Class-Balanced Curriculum.
  [[pdf]](https://arxiv.org/abs/2004.08514) [[code]](https://github.com/voldemortX/DST-CBC) *arXiv 2004.08514*
  
- Evolutionary Population Curriculum for Scaling Multi-Agent Reinforcement Learning.
  [[pdf]](https://arxiv.org/pdf/2003.10423.pdf)[[code]](https://github.com/qian18long/epciclr2020) *ICLR*
  - "Evolutionary Population Curriculum (EPC), a curriculum learning paradigm that scales up MultiAgent Reinforcement Learning (MARL) by progressively increasing the population of training agents in a stage-wise manner."

- Curriculum Loss: Robust Learning and Generalization Against Label Corruption.
  [[pdf]](https://arxiv.org/pdf/1905.10045.pdf) *ICLR*

- Automatic Curriculum Learning through Value Disagreement.
  [[pdf]](https://papers.nips.cc/paper/2020/file/566f0ea4f6c2e947f36795c8f58ba901-Paper.pdf) *NeurIPS*
  - " When biological agents learn, there is often an organized and meaningful order to which learning happens."
  - "Our key insight is that if we can sample goals at the frontier of the set of goals that an agent is able to reach, it will provide a significantly stronger learning signal compared to randomly sampled goals"
<p align="center">
  <img src="https://github.com/Openning07/awesome-curriculum-learning/blob/master/images/AutomaticCurriculumLearningThroughValueDisagreement.png" alt="CMM" width="65%">
</p>

- Curriculum by Smoothing.
  [[pdf]](https://proceedings.neurips.cc/paper/2020/file/f6a673f09493afcd8b129a0bcf1cd5bc-Paper.pdf) [[code]](https://github.com/pairlab/CBS) *NeurIPS*

- Curriculum Learning by Dynamic Instance Hardness.
  [[pdf]](https://papers.nips.cc/paper/2020/file/62000dee5a05a6a71de3a6127a68778a-Paper.pdf) *NeurIPS*

- Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID.
  [[pdf]](https://arxiv.org/pdf/2006.02713.pdf) [[code]](https://github.com/yxgeee/SpCL) [[zhihu]](https://zhuanlan.zhihu.com/p/269112325?utm_source=wechat_session&utm_medium=social&utm_oi=41299705069568&utm_content=group3_article&utm_campaign=shareopn&wechatShare=2&s_r=0) *NeurIPS*

- Self-Paced Deep Reinforcement Learning.
 [[pdf]](https://papers.nips.cc/paper/2020/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html) *NeurIPS*
 
- SuperLoss: A Generic Loss for Robust Curriculum Learning.
  [[pdf]](https://proceedings.neurips.cc/paper/2020/file/2cfa8f9e50e0f510ede9d12338a5f564-Paper.pdf) [[code]](https://github.com/AlanChou/Super-Loss) *NeurIPS*

- Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey.
  [[pdf]](https://jmlr.org/papers/volume21/20-212/20-212.pdf) *JMLR*

### 2021
- Curriculum Labeling: Revisiting Pseudo-Labeling for Semi-Supervised Learning.
  [[pdf]](https://arxiv.org/pdf/2001.06001.pdf) [[code]](https://github.com/uvavision/Curriculum-Labeling) *AAAI*

- Robust Curriculum Learning: from clean label detection to noisy label self-correction.
  [[pdf]](https://openreview.net/pdf?id=lmTWnm3coJJ) [[online review]](https://openreview.net/forum?id=lmTWnm3coJJ) *ICLR*
  - "Robust curriculum learning (RoCL) improves noisy label learning by periodical transitions from supervised learning of clean labeled data to self-supervision of wrongly-labeled data, where the data are selected according to training dynamics."

- Robust Early-Learning: Hindering The Memorization of Noisy Labels.
  [[pdf]](https://openreview.net/pdf?id=Eql5b1_hTE4) [[online review]](https://openreview.net/forum?id=Eql5b1_hTE4) *ICLR*
  - "Robust early-learning: to reduce the side effect of noisy labels before early stopping and thus enhance the memorization of clean labels. Specifically, in each iteration, we divide all parameters into the critical and non-critical ones, and then perform different update rules for different types of parameters."

- When Do Curricula Work?
  [[pdf]](https://openreview.net/pdf?id=tW4QEInpni) *ICLR* (oral)
  - "We find that for standard benchmark datasets, curricula have only marginal benefits, and that randomly ordered samples perform as well or better than curricula and anti-curricula, suggesting that any benefit is entirely due to the dynamic training set size. ... Our experiments demonstrate that curriculum, but not anti-curriculum or random ordering can indeed improve the performance either with limited training time budget or in the existence of noisy data."

- Curriculum Graph Co-Teaching for Multi-Target Domain Adaptation. `TL`
  [[pdf]](https://arxiv.org/pdf/2104.00808.pdf) [[code]](https://github.com/Evgeneus/Graph-Domain-Adaptaion) *CVPR*
<p align="center">
  <img src="https://github.com/Evgeneus/Graph-Domain-Adaptaion/blob/master/data/pipeline.png" alt="Curriculum Graph Co-Teaching" width="80%">
</p>

- Unsupervised Curriculum Domain Adaptation for No-Reference Video Quality Assessment. `Cls`
  [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Unsupervised_Curriculum_Domain_Adaptation_for_No-Reference_Video_Quality_Assessment_ICCV_2021_paper.pdf) [[code]](https://github.com/cpf0079/UCDA) *ICCV*

- Adaptive Curriculum Learning.
  [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Kong_Adaptive_Curriculum_Learning_ICCV_2021_paper.pdf) *ICCV*

- Multi-Level Curriculum for Training A Distortion-Aware Barrel Distortion Rectification Model.
  [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Liao_Multi-Level_Curriculum_for_Training_a_Distortion-Aware_Barrel_Distortion_Rectification_Model_ICCV_2021_paper.pdf) *ICCV*

- TeachMyAgent: a Benchmark for Automatic Curriculum Learning in Deep RL.
  [[pdf]](https://arxiv.org/pdf/2103.09815.pdf) [[code]](https://github.com/flowersteam/TeachMyAgent) *ICML*
<p align="center">
  <img src="https://github.com/flowersteam/TeachMyAgent/blob/master/TeachMyAgent/graphics/readme_graphics/global_schema.png" alt="TechMyAgent" width="52%">
</p>

- Self-Paced Context Evaluation for Contextual Reinforcement Learning.  `RL`
  [[pdf]](https://arxiv.org/pdf/2106.05110.pdf) *ICML*
  - "To improve sample efficiency for learning on such instances of a problem domain, we present Self-Paced Context Evaluation (SPaCE). Based on self-paced learning, \spc automatically generates \task curricula online with little computational overhead. To this end, SPaCE leverages information contained in state values during training to accelerate and improve training performance as well as generalization capabilities to new instances from the same problem domain."

- Curriculum Learning by Optimizing Learning Dynamics.
  [[pdf]](http://proceedings.mlr.press/v130/zhou21a/zhou21a.pdf) [[code]](https://github.com/tianyizhou/DoCL) *AISTATS*
<p align="center">
  <img src="https://github.com/tianyizhou/DoCL/raw/main/docl_aistats2021_thumbnail.png" alt="DoCL" width="52%">
</p>

- FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling. `Cls`
  [[pdf]](https://arxiv.org/pdf/2110.08263.pdf) [[code]](https://github.com/torchssl/torchssl) *NeurIPS*

- Learning with Noisy Correspondence for Cross-modal Matching.
  [[pdf]](https://proceedings.neurips.cc/paper/2021/file/f5e62af885293cf4d511ceef31e61c80-Paper.pdf) [[code]](https://github.com/XLearning-SCU/2021-NeurIPS-NCR) *NeurIPS*
<p align="center">
  <img src="https://github.com/Openning07/awesome-curriculum-learning/blob/master/images/NCR_NeurIPS21.png" alt="NCR" width="85%">
</p>

- Self-Paced Contrastive Learning for Semi-Supervised Medical Image Segmentation with Meta-labels.
  [[pdf]](https://proceedings.neurips.cc/paper/2021/file/8b5c8441a8ff8e151b191c53c1842a38-Paper.pdf) *NeurIPS*
  - "A self-paced learning strategy exploiting the weak annotations is proposed to further help the learning process and discriminate useful labels from noise."

- Curriculum Learning for Vision-and-Language Navigation.
  [[pdf]](https://proceedings.neurips.cc/paper/2021/file/6f0442558302a6ededff195daf67f79b-Paper.pdf) *NeurIPS*
  - "We propose a novel curriculum- based training paradigm for VLN tasks that can balance human prior knowledge and agent learning progress about training samples."

### 2022
- Characterizing and overcoming the greedy nature of learning in multi-modal deep neural networks.
  [[pdf]](https://arxiv.org/pdf/2202.05306.pdf) [[code]](https://github.com/nyukat/greedy_multimodal_learning) *ICML*
  - "We hypothesize that due to the greedy nature of learning in multi-modal deep neural networks, these models tend to rely on just one modality while under-fitting the other modalities. ... We propose an algorithm to balance the conditional learning speeds between modalities during training and demonstrate that it indeed addresses the issue of greedy learning."

- Pseudo-Labeled Auto-Curriculum Learning for Semi-Supervised Keypoint Localization.  `Seg`
  [[pdf]](https://arxiv.org/pdf/2201.08613.pdf) [[open review]](https://openreview.net/forum?id=6Q52pZ-Th7N) *ICLR*
  - "We propose to automatically select reliable pseudo-labeled samples with a series of dynamic thresholds, which constitutes a learning curriculum."

- C-Planning: An Automatic Curriculum for Learning Goal-Reaching Tasks. `RL`
  [[pdf]](https://openreview.net/pdf?id=K2JfSnLBD9) [[open review]](https://openreview.net/forum?id=K2JfSnLBD9) *ICLR*

- Curriculum learning as a tool to uncover learning principles in the brain.
  [[pdf]](https://openreview.net/pdf?id=TpJMvo0_pu-) [[open review]](https://openreview.net/forum?id=TpJMvo0_pu-) *ICLR*
  
- It Takes Four to Tango: Multiagent Self Play for Automatic Curriculum Generation.
  [[pdf]](https://openreview.net/pdf?id=q4tZR1Y-UIs) [[open review]](https://openreview.net/forum?id=q4tZR1Y-UIs) *ICLR*
  
- Boosted Curriculum Reinforcement Learning. `RL`
  [[pdf]](https://openreview.net/pdf?id=anbBFlX1tJ1) [[open review]](https://openreview.net/forum?id=anbBFlX1tJ1) *ICLR*

- ST++: Make Self-Training Work Better for Semi-Supervised Semantic Segmentation.
  [[pdf]](https://arxiv.org/pdf/2106.05095.pdf) [[code]](https://github.com/LiheYoung/ST-PlusPlus) *CVPR*
  - "propose a selective re-training scheme via prioritizing reliable unlabeled samples to safely exploit the whole unlabeled set in an easy-to-hard curriculum learning manner."

- Robust Cross-Modal Representation Learning with Progressive Self-Distillation.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Andonian_Robust_Cross-Modal_Representation_Learning_With_Progressive_Self-Distillation_CVPR_2022_paper.pdf) *CVPR*
  - "The learning objective of vision-language approach of CLIP does not effectively account for the noisy many-to-many correspondences found in web-harvested image captioning datasets. To address this challenge, we introduce a novel training framework based on cross-modal contrastive learning that uses progressive self-distillation and soft image-text alignments to more efficiently learn robust representations from noisy data."
