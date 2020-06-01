# ScanSSD: Scanning Single Shot Detector for Math in Document Images

A [PyTorch](http://pytorch.org/) implementation of ScanSSD [Scanning Single Shot MultiBox Detector](https://paragmali.me/scanning-single-shot-detector-for-math-in-document-images/) by [**Parag Mali**](https://github.com/MaliParag/). It was developed using SSD implementation by [**Max deGroot**](https://github.com/amdegroot).

Developed using Cuda 9.1.85 and Pytorch 1.1.0

<img align="right" src=
"https://github.com/maliparag/scanssd/blob/master/images/detailed_math512_arch.png" height = 400/>

&nbsp;
&nbsp;

## Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#code-organization'>Code Organization</a>
- <a href='#training-scanssd'>Training</a>
- <a href='#testing'>Testing</a>
- <a href='#performance'>Performance</a>

&nbsp;
&nbsp;

## Installation
- Clone this repository. Requires Python 3
- Install [PyTorch](http://pytorch.org/)
- Download the dataset by following the instructions at https://github.com/MaliParag/TFD-ICDAR2019
- Install [Visdom](https://github.com/facebookresearch/visdom) to see real-time loss visualization during training
  * To use Visdom in a web browser:
  ```Shell
  # First install Python server and client
  pip install visdom
  # Start the server (probably in a screen or tmux)
  python -m visdom.server
  ```
  * Then (during training) navigate to http://localhost:8097/ (see the Train section below for training details).

## Code Organization
 
SSD model is built in `ssd.py`. Training and testing the SSD is managed in `train.py` and `test.py`. All the training code is in the `layers` directory. Hyper-parameters for training and testing can be specified through the command line and through `config.py` inside the `data` directory. 

The `data` directory also contains `gtdb_new.py` which is a data reader that uses sliding windows to generates sub-images of page for training. All the scripts regarding stitching the sub-image level detections are in the `gtdb` directory. 

Functions for data augmentation, visualization of bounding boxes and heatmap are in the `utils` directory. 

## Setting up data for training

If you are not sure how to setup data, look at the [dir_struct](https://github.com/MaliParag/ScanSSD/blob/master/dir_struct) file. It has a possible directory structure that you can use for setting up data for training and testing. This repository has been organized in accordance with the structure outlined in that file. 

To generate .pmath files or .pchar files you can use [this](https://github.com/MaliParag/ScanSSD/blob/master/gtdb/split_annotations_per_page.py) script. 

## Pre-Trained Weights

For quick testing, pre-trained weights are available [here.](https://drive.google.com/file/d/1bGNvg9uLCTbVE9hk1yWE-2tLgX1eg_me/view?usp=sharing) A copy is included in this repository in the `ssd` directory. 

## Training ScanSSD Weights

- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights [here](https://drive.google.com/file/d/1GqiyZ1TglNW5GrNQfXQ72S8mChhJ4_sD/view?usp=sharing)
- By default, we assume you have downloaded the file in the `ssd/base_weights` directory
- Run the following command (example run from the `ssd` directory):

```Shell
python3 train.py 
--dataset GTDB 
--dataset_root ../
--cuda True 
--visdom True 
--batch_size 16 
--num_workers 4 
--exp_name IOU512_iter1 
--model_type 512 
--training_data training_data 
--cfg hboxes512 
--loss_fun ce 
--kernel 1 5 
--padding 0 2 
--neg_mining True 
--pos_thresh 0.75
```

What each command line argument means (see `train.py` for more optional arguments):
```Shell
python3 train.py 
--dataset       # the name of the dataset
--dataset_root  # the home/root directory of the dataset/project
--cuda          # T/F, use NVIDIA card if True
--visdom        # T/F, running visdom if True
--batch_size    # training batch size
--num_workers   # number of workers for data loading
--exp_name      # experiment name - prefix for files generated
--model_type    # type of SSD model; ssd300 or ssd512
--training_data # the training data to use
--cfg           # type of network; gtdb, math_gtdb_512 or hboxes512
--loss_fun      # type of loss; fl (focal loss) or ce (cross entropy)
--kernel        # kernel size for feature layers; 3 3 or 1 5
--padding       # padding for feature layers; 1 1 or 0 2
--neg_mining    # T/F, use hard negative mining with ratio 1:3 if True
--pos_thresh    # all default boxes with IOU > pos_thresh are
                #       considered positive examples
```

Notes:
  - For training, an NVIDIA GPU is strongly recommended for speed.
  - For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section.
  - A trained weight file is saved every 1,000 training cycles.
  - You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train.py` for options)

## Testing

To test a trained network (example run from the `ssd` directory):

```Shell
python3 test.py 
--dataset_root ../ 
--trained_model HBOXES512_iter1GTDB.pth  
--visual_threshold 0.25 
--cuda True 
--exp_name test_real_world_iter1 
--test_data testing_data  
--model_type 512 
--cfg hboxes512 
--padding 3 3 
--kernel 1 5 
--batch_size 8
```

What each command line argument means (see `test.py` for more optional arguments):

```Shell
python3 test.py 
--dataset_root      # the home/root directory of the dataset/project
--trained_model     # the trained weight file (including file path and name)
--visual_threshold  # the final confidence threshold
--cuda              # T/F, use NVIDIA card if True
--exp_name          # experiment name - used to create directory where
                    #       results are stored
--test_data         # the testing data file. Each line in the file specifies
                    #       one page image as document_name/page_number
--model_type        # type of SSD model; ssd300 or ssd512
--cfg               # type of network; gtdb, math_gtdb_512 or hboxes512
--padding           # padding for feature layers; 1 1 or 0 2
--kernel            # kernel size for feature layers; 3 3 or 1 5
--batch_size        # testing batch size
```

Notes:
- Training outputs .csv files for each document. These are saved in a directory created in the `eval` 
directory with the name specified by the `--exp_name` command line argument.
- You can specify the parameters listed in the `eval.py` file by flagging them or manually changing them.  

## Stitching the patch level results

- Used to stitch the results from `test.py` back into full pages

```Shell
python3 <Workspace>/ssd/gtdb/stitch_patches_pdf.py 
--data_file <Workspace>/train_pdf 
--output_dir <Workspace>/ssd/eval/stitched_HBOXES512_e4/ 
--math_dir <Workspace>/ssd/eval/test_HBOXES512_e4/ 
--stitching_algo equal 
--algo_threshold 30 
--num_workers 8 
--postprocess True 
--home_images <Workspace>/images/ 
```

What each command line argument means:

```Shell
python3 <Workspace>/ssd/gtdb/stitch_patches_pdf.py 
--data_file         # training data; list of filenames, one per line
--output_dir        # directory to store outputs in
--math_dir          # directory where test.py output was stored
--stitching_algo    # what stitching algorithm to use
--algo_threshold    # stitching algorithm threshold
--num_workers       # number of workers used in data loading
--postprocess       # T/F, fit math regions before pooling if True
--home_images       # the directory of the original document images
```

## Visualization
- Use `visualize_annotations.py` to visualize the results (example run from the `ssd` directory):

```Shell
python3 <Workspace>/ICDAR2019/TFD-ICDAR2019v2/VisualizationTools/visualize_annotations.py
--img_dir ../images/
--out_dir visual/HBOXES512_e4_data/
--math_dir eval/stitched_HBOXES512_e4/
```

What each command line argument means:

```Shell
python3 <Workspace>/ICDAR2019/TFD-ICDAR2019v2/VisualizationTools/visualize_annotations.py
--img_dir   # directory that contains the original images 
--out_dir   # directory to store visualization output in
--math_dir  # directory where stitching output was stored
```

## Evaluate 

```
python3 <Workspace>/ICDAR2019/TFD-ICDAR2019v2/Evaluation/IOULib/IOUevaluater.py 
--ground_truth <Workspace>/ICDAR2019/TFD-ICDAR2019v2/Test/gt/math_gt/ 
--detections <Workspace>/ssd/eval/stitched_HBOXES512_e4/
```

## Performance

#### TFD-ICDAR 2019 Version1 Test

| Metric | Precision | Recall | F-score |
|:-:|:-:|:-:|:-:|
| IOU50 | 85.05 % | 75.85% | 80.19% |
| IOU75 | 77.38 % | 69.01% | 72.96% |

##### FPS
**GTX 1080:** ~27 FPS for 512 * 512 input images

## Related publications

Mali, Parag, et al. “ScanSSD: Scanning Single Shot Detector for Mathematical Formulas in PDF Document Images.” ArXiv:2003.08005 [Cs], Mar. 2020. arXiv.org, http://arxiv.org/abs/2003.08005.

P. S. Mali, ["Scanning Single Shot Detector for Math in Document Images."](https://scholarworks.rit.edu/theses/10210/) Order No. 22622391, Rochester Institute of Technology, Ann Arbor, 2019.

M. Mahdavi, R. Zanibbi, H. Mouchere, and Utpal Garain (2019). [ICDAR 2019 CROHME + TFD: Competition on Recognition of Handwritten Mathematical Expressions and Typeset Formula Detection.](https://www.cs.rit.edu/~rlaz/files/CROHME+TFD%E2%80%932019.pdf) Proc. International Conference on Document Analysis and Recognition, Sydney, Australia (to appear).

## Acknowledgements
- [**Max deGroot**](https://github.com/amdegroot) for providing open-source SSD code
