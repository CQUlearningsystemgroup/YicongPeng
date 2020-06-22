## MoNet3D: Towards Accurate Monocular 3D Object Localization in Real Time


<img src="https://raw.githubusercontent.com/CQUlearningsystemgroup/YicongPeng/master/demo.jpg">

Prerequisites
- Ubuntu 16.04
- Python 3.6
- Tensorflow 1.10.0

Install

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
Evaluation
Run the evaluating script:
```bash
python eval.py
```
Evaluation results have been put on the "./val_out/val_out",if you want to compute mAp,
```bash
./submodules/KittiEvaluation/evaluate_object /path/to/prediction /path/to/gt
```
