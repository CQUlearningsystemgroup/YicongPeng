# MoNet3D: Towards Accurate Monocular 3D Object Localization in Real Time
## Answers to Comments

### Reviewer #2
Overall Comments: The motivation of utilizing local geometry restriction is interesting.

Question 1: The experiments are not convincing and do not validate the effectiveness of the proposed regularization term.
Answers: Thank you for the good suggestions. A group of experiments were added as requested.

[Ablation test] Suppose lambda is the weight of regularization. When lambda=0 (no regularization) the average 3D detection accuracy rates are 13.88%, 10.19%, 7.62% respectively over the Easy, Mod and Hard KITTI sets (IoU = 0.7), which are 6.54%-8.85% LOWER than the results achieved with lambda=100. Moreover, the false positive rate dropped by 18.72% due to the proposed regularization, which is important for ADAS application. Note that the experiment setting and hyper-parameters are the SAME as the compared research (Qin et al, AAAI 2019) in the ablation test.

[Qualitative results] Experiment shows that, the detection accuracy is robust with different values of lambda in the range between 10(16.66%, IoU= 0.7, Mod) to 100(16.73%, IoU= 0.7, Mod).

[Hyper-parameters] The values of hyper-parameters in our experiments were not fine-tuned, e.g. α=10, γ=10, which were the same as (Qin et al, AAAI2019) for fair comparison.

Question 2: The authors should provide the performances on the test set of KITTI.
Answers: Thank you for the good question. The main reason is because the KITTI website does not provide the labels of the official KITTI test set; Therefore, same as the setting of compared researches (Chen et.al, NIPS2015; Li et.al, CVPR2019), we split the KITTI set of 7481 images evenly into two subsets for training (3712 images) and testing(3769 images). No validation images were used because we didn’t tune the hyper-parameters. The misuse of the term “KITTI validation set” will be revised to avoid misunderstanding.

### Reviewer #5
Overall Comments: Very good paper, I would like to see it accepted.

Question 1. It would be useful to clearly and early on mention the exact architecture of the feature extractor.
Answers: Thanks. We will revise the paper accordingly.

Question 2. Equations 4 and 5 could be presented more clearly. Additionally, the need for the term 'p' in Equation 5 is unclear.
Answers: Sorry to say that the term p is a typo, which has been revised. We will put a brief explanation of Eq. 4 and 5 in the revised paper.

Question 3. In section 4.2, it is not explained why only estimation in the horizontal direction for M3D is emphasized.
Answers: Two reasons 1) the crucial task of lane determination depends on the horizontal localization result. 2) the horizontal measurements are more reliable than the depth measurements on 2D images.

Question 4. A small explanation on how instance level depth was compared with pixel level depth could be useful.
Answers: The task of instance depth estimation, which is useful for ADAS application, is less sophisticated than pixel level depth estimation; therefore it leads to higher accuracy and more efficient computation.

Question 5. In Section 4.2, when describing object detection accuracy, an IOU of 0.3 in 3D object detection experiments seems to be a rather low threshold.
Answers: Our paper also compared the results of IoU 0.5 and 0.7 in the experiment. The result of IoU 0.3 may be more meaningful for ADAS application than for automated driving which requires much higher accuracy.

### Reviewer #6
Overall Comments: Very good paper. The method got good excellent results - with fantastic scope of real-world applications.

Question 1. The title of your paper talks about "objects" - well, what about other outdoor and indoor objects + scenes? How well will this adapt and work too?
Answers: Thank you for your suggestion. Though the ICML paper mainly focused on cars, but in theory, the MoNet3D could be used to detect other rigid or non-rigid objects as long as the cameras are mounted with the same angle and configurations. We hope our code and work may benefit all object detection researchers.

Question 2.The code is well structured and well documented，but could not be run on Ubuntu 14.04.
Answer：Thanks for reminding, we will update the code when the paper is published.

Question 3. Your references are not consistent (typo). And it is better to state once what are the red contours (iso-contour lines) in fig. 3 (also 2).
Answer: Thanks, we will revise the paper accordingly.

Question 4. The last stage of fig 2 is not clear. Are Eqns. (7-8) not explicitly specified. Where is it used finally? What is the algo/framework used for minimization?
Answer: The final calculation in Fig 2 is to transform the predictions into the KITTI format. Eq 7 is the loss function of the center point of the 3D frame of the object. Eq 8 is the loss function of the vertices of the 3D frame. The training algorithm is the SGD algorithm with momentum. We will briefly clarify these points in this article.


## Introuduction of the demo

<img src="https://raw.githubusercontent.com/CQUlearningsystemgroup/YicongPeng/master/demo.jpg">

### Prerequisites
- Ubuntu 16.04
- Python 3.6
- Tensorflow 1.10.0

### Install

Download the [Kitti Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) ([image](http://www.cvlibs.net/download.php?file=data_object_image_2.zip), [calib](http://www.cvlibs.net/download.php?file=data_object_calib.zip) and [label](http://www.cvlibs.net/download.php?file=data_object_label_2.zip)) and place it into `data/KittiBox`. 
The train-val split `train.txt` and `val.txt` are contained in this repository.
 
Compile the Cython module and download the pretrained model:
```bash
python setup.py
```
Download the .ckpt files and eval results from https://pan.baidu.com/s/1D5exT_8dt-xzWau5cDPTlA (Extracted code：z6a2),and put them in `ckpt`

The folder should be in the following structure:
```
data
    KittiBox
        training
            calib
            image_2
            label_2
        train.txt
        val.txt
    model_2D.pkl
    model_3D.data-00000-of-00001
    model_3D.index
    model_3D.meta
```
### Evaluation
Run the evaluating script:
```bash
python eval.py
```
Evaluation results have been put on the "./val_out/val_out",if you want to compute mAp,
```bash
./submodules/KittiEvaluation/evaluate_object /path/to/prediction /path/to/gt
```
