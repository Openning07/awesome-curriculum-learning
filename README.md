# Awesome Curriculum Learning[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
Some bravo or inspiring research works on the topic of curriculum learning.

A curated list of awesome Curriculum Learning resources. Inspired by [awesome-deep-vision](https://github.com/kjw0612/awesome-deep-vision), [awesome-adversarial-machine-learning](https://github.com/yenchenlin/awesome-adversarial-machine-learning), [awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers), and [awesome-architecture-search](https://github.com/markdtw/awesome-architecture-search)

#### Why Curriculum Learning?
Self-Supervised Learning has become an exciting direction in AI community. 
  - Bengio: "Supervision is the opium of the AI researcher" (ICML 2009)

## Contributing
<p align="center">
  <img src="http://cdn1.sportngin.com/attachments/news_article/7269/5172/needyou_small.jpg" alt="We Need You!">
</p>

Please help contribute this list by contacting [me](https://github.com/Openning07/awesome-curriculum-learning) or add [pull request](https://github.com/Openning07/awesome-curriculum-learning/pulls)

Markdown format:
```markdown
- Paper Name. 
  [[pdf]](link) 
  [[code]](link)
  - Author 1, Author 2, and Author 3. *Conference Year*
```

## Table of Contents
- [Computer Vision (CV)](#computer-vision)
  - [Survey](#survey)
  - [Image Representation Learning](#image-representation-learning)

## Computer Vision
### Survey
- Self-supervised Visual Feature Learning with Deep Neural Networks: A Survey.
  [[pdf]](https://arxiv.org/pdf/1902.06162.pdf)
  - Longlong Jing and Yingli Tian.

### Image Representation Learning

#### Benchmark code
FAIR Self-Supervision Benchmark [[repo]](https://github.com/facebookresearch/fair_self_supervision_benchmark): various benchmark (and legacy) tasks for evaluating quality of visual representations learned by various self-supervision approaches. 

#### 2015
- Unsupervised Visual Representation Learning by Context Prediction.
  [[pdf]](https://arxiv.org/abs/1505.05192)
  [[code]](http://graphics.cs.cmu.edu/projects/deepContext/)
  - Doersch, Carl and Gupta, Abhinav and Efros, Alexei A. *ICCV 2015*

- Unsupervised Learning of Visual Representations using Videos.
  [[pdf]](http://www.cs.cmu.edu/~xiaolonw/papers/unsupervised_video.pdf) 
  [[code]](http://www.cs.cmu.edu/~xiaolonw/unsupervise.html)
  - Wang, Xiaolong and Gupta, Abhinav. *ICCV 2015*

- Learning to See by Moving. 
  [[pdf]](http://arxiv.org/abs/1505.01596)
  [[code]](https://people.eecs.berkeley.edu/~pulkitag/lsm/lsm.html)
  - Agrawal, Pulkit and Carreira, Joao and Malik, Jitendra. *ICCV 2015*

- Learning image representations tied to ego-motion.
  [[pdf]](http://vision.cs.utexas.edu/projects/egoequiv/ijcv_bestpaper_specialissue_egoequiv.pdf) 
  [[code]](http://vision.cs.utexas.edu/projects/egoequiv/)
  - Jayaraman, Dinesh and Grauman, Kristen. *ICCV 2015*

#### 2016
- Joint Unsupervised Learning of Deep Representations and Image Clusters. 
  [[pdf]](https://arxiv.org/pdf/1604.03628.pdf) 
  [[code-torch]](https://github.com/jwyang/JULE.torch)
  [[code-caffe]](https://github.com/jwyang/JULE-Caffe)
  - Jianwei Yang, Devi Parikh, Dhruv Batra. *CVPR 2016*
  
- Unsupervised Deep Embedding for Clustering Analysis.
  [[pdf]](https://arxiv.org/pdf/1511.06335.pdf) 
  [[code]](https://github.com/piiswrong/dec)
  - Junyuan Xie, Ross Girshick, and Ali Farhadi. *ICML 2016*
  
- Slow and steady feature analysis: higher order temporal coherence in video. 
  [[pdf]](http://vision.cs.utexas.edu/projects/slowsteady/cvpr16.pdf)
  - Jayaraman, Dinesh and Grauman, Kristen. *CVPR 2016*

- Context Encoders: Feature Learning by Inpainting. 
  [[pdf]](https://people.eecs.berkeley.edu/~pathak/papers/cvpr16.pdf)
  [[code]](https://people.eecs.berkeley.edu/~pathak/context_encoder/)
  - Pathak, Deepak and  Krahenbuhl, Philipp and Donahue, Jeff and Darrell, Trevor and Efros, Alexei A. *CVPR 2016*

- Colorful Image Colorization.
  [[pdf]](https://arxiv.org/abs/1603.08511)
  [[code]](http://richzhang.github.io/colorization/)
  - Zhang, Richard and Isola, Phillip and Efros, Alexei A. *ECCV 2016*

- Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles.
  [[pdf]](http://arxiv.org/abs/1603.09246)
  [[code]](http://www.cvg.unibe.ch/research/JigsawPuzzleSolver.html)
  - Noroozi, Mehdi and Favaro, Paolo. *ECCV 2016*

- Ambient Sound Provides Supervision for Visual Learning.
  [[pdf]](http://arxiv.org/pdf/1608.07017) 
  [[code]](http://andrewowens.com/ambient/index.html)
  - Owens, Andrew and Wu, Jiajun and McDermott, Josh and Freeman, William and Torralba, Antonio. *ECCV 2016*

- Learning Representations for Automatic Colorization. 
  [[pdf]](http://arxiv.org/pdf/1603.06668.pdf)
  [[code]](http://people.cs.uchicago.edu/~larsson/colorization/)
  - Larsson, Gustav and Maire, Michael and Shakhnarovich, Gregory. *ECCV 2016*

-   Unsupervised Visual Representation Learning by Graph-based Consistent Constraints.
    [\[pdf\]](http://faculty.ucmerced.edu/mhyang/papers/eccv16_feature_learning.pdf)
    [\[code\]](https://github.com/dongli12/FeatureLearning)
    -   Li, Dong and Hung, Wei-Chih and Huang, Jia-Bin and Wang, Shengjin and Ahuja, Narendra and Yang, Ming-Hsuan. *ECCV 2016*

#### 2017
- Adversarial Feature Learning. 
  [[pdf]](https://arxiv.org/pdf/1605.09782.pdf)
  [[code]](https://github.com/jeffdonahue/bigan)
  - Donahue, Jeff and Krahenbuhl, Philipp and Darrell, Trevor. *ICLR 2017*
  
- Self-supervised learning of visual features through embedding images into text topic spaces.
  [[pdf]](https://arxiv.org/pdf/1705.08631.pdf)
  [[code]](https://github.com/lluisgomez/TextTopicNet)
  - L. Gomez* and Y. Patel* and M. Rusiñol and D. Karatzas and C.V. Jawahar. *CVPR 2017*
  
- Split-Brain Autoencoders: Unsupervised Learning by Cross-Channel Prediction.
  [[pdf]](https://arxiv.org/abs/1611.09842) 
  [[code]](https://github.com/richzhang/splitbrainauto)
  - Zhang, Richard and Isola, Phillip and Efros, Alexei A. *CVPR 2017*

- Learning Features by Watching Objects Move.
  [[pdf]](https://people.eecs.berkeley.edu/~pathak/papers/cvpr17.pdf) 
  [[code]](https://people.eecs.berkeley.edu/~pathak/unsupervised_video/)
  - Pathak, Deepak and Girshick, Ross and Dollar, Piotr and  Darrell, Trevor and Hariharan, Bharath. *CVPR 2017*
  
- Colorization as a Proxy Task for Visual Understanding. 
  [[pdf]](http://arxiv.org/abs/1703.04044) 
  [[code]](http://people.cs.uchicago.edu/~larsson/color-proxy/)
  - Larsson, Gustav and Maire, Michael and Shakhnarovich, Gregory. *CVPR 2017*

-   DeepPermNet: Visual Permutation Learning.
    [\[pdf\]](https://arxiv.org/pdf/1704.02729.pdf)
    [\[code\]](https://github.com/rfsantacruz/deep-perm-net)
    -   Cruz, Rodrigo Santa and Fernando, Basura and Cherian, Anoop and Gould, Stephen. *CVPR 2017*

- Unsupervised Learning by Predicting Noise.
  [[pdf]](https://arxiv.org/abs/1704.05310) 
  [[code]](https://github.com/facebookresearch/noise-as-targets)
  - Bojanowski, Piotr and Joulin, Armand. *ICML 2017*

- Multi-task Self-Supervised Visual Learning. 
  [[pdf]](https://arxiv.org/abs/1708.07860)
  - Doersch, Carl and Zisserman, Andrew. *ICCV 2017*

- Representation Learning by Learning to Count.
  [[pdf]](https://arxiv.org/abs/1708.06734)
  - Noroozi, Mehdi and Pirsiavash, Hamed and Favaro, Paolo. *ICCV 2017*

- Transitive Invariance for Self-supervised Visual Representation Learning.
  [[pdf]](https://arxiv.org/pdf/1708.02901.pdf)
  - Wang, Xiaolong and He, Kaiming and Gupta, Abhinav. *ICCV 2017*

- Look, Listen and Learn. 
  [[pdf]](https://arxiv.org/pdf/1705.08168.pdf)
  - Relja, Arandjelovic and Zisserman, Andrew. *ICCV 2017*

- Unsupervised Representation Learning by Sorting Sequences. 
  [[pdf]](https://arxiv.org/pdf/1708.01246.pdf) 
  [[code]](https://github.com/HsinYingLee/OPN)
  - Hsin-Ying Lee, Jia-Bin Huang, Maneesh Kumar Singh, and Ming-Hsuan Yang. *ICCV 2017*

#### 2018

- Unsupervised Feature Learning via Non-parameteric Instance Discrimination
  [[pdf]](https://arxiv.org/pdf/1805.01978.pdf) 
  [[code]](https://github.com/zhirongw/lemniscate.pytorch)
  - Zhirong Wu, Yuanjun Xiong and X Yu Stella and Dahua Lin. *CVPR 2018*

- Learning Image Representations by Completing Damaged Jigsaw Puzzles. 
  [[pdf]](https://arxiv.org/pdf/1802.01880.pdf)
  - Kim, Dahun and Cho, Donghyeon and Yoo, Donggeun and Kweon, In So. *WACV 2018*
        
- Unsupervised Representation Learning by Predicting Image Rotations. 
  [[pdf]](https://openreview.net/forum?id=S1v4N2l0-)
  [[code]](https://github.com/gidariss/FeatureLearningRotNet)
  - Spyros Gidaris and Praveer Singh and Nikos Komodakis. *ICLR 2018*
  
- Improvements to context based self-supervised learning. 
  [[pdf]](https://arxiv.org/abs/1711.06379)
  - Terrell Mundhenk and Daniel Ho and Barry Chen. *CVPR 2018*
  
- Self-Supervised Feature Learning by Learning to Spot Artifacts.
  [[pdf]](https://arxiv.org/pdf/1806.05024.pdf)
  [[code]](https://github.com/sjenni/LearningToSpotArtifacts)
  - Simon Jenni and Universität Bern and Paolo Favaro. *CVPR 2018*
  
- Boosting Self-Supervised Learning via Knowledge Transfer. 
  [[pdf]](https://www.csee.umbc.edu/~hpirsiav/papers/transfer_cvpr18.pdf)
  - Mehdi Noroozi and Ananth Vinjimoor and Paolo Favaro and Hamed Pirsiavash. *CVPR 2018*
  
- Cross-domain Self-supervised Multi-task Feature Learning Using Synthetic Imagery. 
  [[pdf]](https://arxiv.org/abs/1711.09082)
  [[code]](https://github.com/jason718/game-feature-learning)
  - Zhongzheng Ren and Yong Jae Lee. *CVPR 2018*
  
- ShapeCodes: Self-Supervised Feature Learning by Lifting Views to Viewgrids.
  [[pdf]](https://arxiv.org/pdf/1709.00505.pdf)
  - Dinesh Jayaraman*, UC Berkeley; Ruohan Gao, University of Texas at Austin; Kristen Grauman. *ECCV 2018*

- Deep Clustering for Unsupervised Learning of Visual Features
    [[pdf]](https://research.fb.com/wp-content/uploads/2018/09/Deep-Clustering-for-Unsupervised-Learning-of-Visual-Features.pdf)
    - Mathilde Caron, Piotr Bojanowski, Armand Joulin, Matthijs Douze. *ECCV 2018*

- Cross Pixel Optical-Flow Similarity for Self-Supervised Learning.
  [[pdf]](http://www.robots.ox.ac.uk/~vgg/publications/2018/Mahendran18/mahendran18.pdf)
  - Aravindh Mahendran, James Thewlis, Andrea Vedaldi. *ACCV 2018*
 
#### 2019

- Representation Learning with Contrastive Predictive Coding.
  [[pdf]](https://arxiv.org/abs/1807.03748)
  - Aaron van den Oord, Yazhe Li, Oriol Vinyals.

- Self-Supervised Learning via Conditional Motion Propagation.
  [[pdf]](http://www.robots.ox.ac.uk/~vgg/publications/2018/Mahendran18/mahendran18.pdf)
  [[code]](https://github.com/XiaohangZhan/conditional-motion-propagation)
  - Xiaohang Zhan, Xingang Pan, Ziwei Liu, Dahua Lin, and Chen Change Loy. *CVPR 2019*
 
- Self-Supervised Representation Learning by Rotation Feature Decoupling.
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/html/Feng_Self-Supervised_Representation_Learning_by_Rotation_Feature_Decoupling_CVPR_2019_paper.html)
  [[code]](https://github.com/philiptheother/FeatureDecoupling)
  - Zeyu Feng; Chang Xu; Dacheng Tao. *CVPR 2019*
 
- Revisiting Self-Supervised Visual Representation Learning.
  [[pdf]](https://arxiv.org/abs/1901.09005)
  [[code]](https://github.com/google/revisiting-self-supervised)
  - Alexander Kolesnikov; Xiaohua Zhai; Lucas Beye. CVPR 2019

- AET vs. AED: Unsupervised Representation Learning by Auto-Encoding Transformations rather than Data.
  [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_AET_vs._AED_Unsupervised_Representation_Learning_by_Auto-Encoding_Transformations_Rather_CVPR_2019_paper.pdf)
  [[code]](https://github.com/maple-research-lab/AET)
  - Liheng Zhang, Guo-Jun Qi, Liqiang Wang, Jiebo Luo. *CVPR 2019*

- Unsupervised Deep Learning by Neighbourhood Discovery.
  [[pdf]](http://proceedings.mlr.press/v97/huang19b.html).
  [[code]](https://github.com/Raymond-sci/AND).
  - Jiabo Huang, Qi Dong, Shaogang Gong, Xiatian Zhu. *ICML 2019*
  
- Contrastive Multiview Coding.
  [[pdf]](https://arxiv.org/abs/1906.05849)
  [[code]](https://github.com/HobbitLong/CMC/)
  - Yonglong Tian and Dilip Krishnan and Phillip Isola.
 
- Large Scale Adversarial Representation Learning.
  [[pdf]](https://arxiv.org/abs/1907.02544)
  - Jeff Donahue, Karen Simonyan.
 
- Learning Representations by Maximizing Mutual Information Across Views.
  [[pdf]](https://arxiv.org/pdf/1906.00910)
  - Philip Bachman, R Devon Hjelm, William Buchwalter
 
 - Selfie: Self-supervised Pretraining for Image Embedding. 
    [[pdf]](https://arxiv.org/abs/1906.02940) 
    - Trieu H. Trinh, Minh-Thang Luong, Quoc V. Le
   
 - Data-Efficient Image Recognition with Contrastive Predictive Coding
    [[pdf]](https://arxiv.org/abs/1905.09272)
    - Olivier J. He ́naff, Ali Razavi, Carl Doersch, S. M. Ali Eslami, Aaron van den Oord

 - Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty
    [[pdf]](https://arxiv.org/pdf/1906.12340)
    [[code]](https://github.com/hendrycks/ss-ood)
    - Dan Hendrycks, Mantas Mazeika, Saurav Kadavath, Dawn Song. *NeurIPS 2019*
