## Interactive segmentation for avalanches
This repository provides the source code for the click-based interactive segmentation model for the following paper

> **Interactive Snow Avalanche Segmentation from Webcam Imagery: results, potential and limitations**<br>
> Hafner, E. D., Kontogianni, T., Caye Daudt, R., Oberson, L., Wegner, J. D., Schindler, K., and Bühler, Y., EGUsphere [preprint],<br>
> https://doi.org/10.5194/egusphere-2024-498, 2024.
>
the code that served as a basis was initially developed by 
> **Konstantin Sofiiuk, Ilia Petrov, Anton Konushin**<br>
> Samsung Research

and published in the following paper and repository 

> Reviving Iterative Training with Mask Guidance for Interactive Segmentation, https://arxiv.org/abs/2102.06583<br>
> https://github.com/SamsungLabs/ritm_interactive_segmentation


## Setting up the environment

This framework is built using Python 3.6 and relies on the PyTorch 1.4.0+. The following command installs all 
necessary packages:

```.bash
pip3 install -r requirements.txt
```

If you want to run training or testing, you must configure the paths to the datasets in [config.yml](config.yml).

## Interactive Segmentation Demo

<p align="center">
  <img src="./assets/img/demo_gui.jpg" alt="drawing" width="99%"/>
</p>

The GUI is based on TkInter library and its Python bindings. You can try to interactivly segment avalanches with the demo with one of the 
[provided models](#pretrained-models). The scripts automatically detect the architecture of the loaded model, you only need to 
specify the path to the corresponding checkpoint.

Examples of the script usage:

```.bash
# This command runs interactive demo with HRNet18 ITER-M model trained on the SLF dataset from /data/ritm_interactive_segmentation/datasets/checkpoints/
# If you also do not have a lot of GPU memory, you can reduce --limit-longest-size (default=800; 3600 was the longest we could handle with a NVIDIA GeForce RTX 2080 Ti)
python3 demo.py --checkpoint=/data/ritm_interactive_segmentation/datasets/checkpoints/082_epo090.pth --gpu=0 --limit-longest-size=3600

# You can try the demo in CPU only mode
python3 demo.py --checkpoint=/data/ritm_interactive_segmentation/datasets/checkpoints/082_epo090.pth --cpu
```

**Controls**:

| Key                                                           | Description                        |
| ------------------------------------------------------------- | ---------------------------------- |
| <kbd>Left Mouse Button</kbd>                                  | Place a positive click             |
| <kbd>Right Mouse Button</kbd>                                 | Place a negative click             |
| <kbd>Roll scroll Wheel</kbd>                                  | Zoom an image in and out           |
| <kbd> Push scroll Wheel</kbd>                                 | Pan the image                      |
| <kbd>Left Mouse Button</kbd> + <br> <kbd>Move Mouse</kbd>     | Create a bounding Box              |
| <kbd>Space</kbd>                                              | Finish the current object mask     |


**Correct existing external segmentation mask**:

A user can initialize the model with an external mask before placing any clicks and correcting the mask using the same interface. 
To do so the demo can be run with any model train configuration and an external mask can be added via the "Load mask" button in the menu bar and consequently adapted.


**GUI visualization parameters**:
        <li><i>Prediction threshold</i> slider adjusts the threshold for binarization of probability map for the current object.</li> 
        <li><i>Alpha blending coefficient</i> slider adjusts the intensity of all predicted masks.</li>
        <li><i>Visualisation click radius</i> slider adjusts the size of red and green dots depicting clicks.</li>



## Datasets

We train all our models on SBD and COCO+LVIS and evaluate them on GrabCut, Berkeley, DAVIS, SBD and PascalVOC. We also provide links to additional datasets: ADE20k and OpenImages, that are used in ablation study.

| Dataset   |                      Description             |           Download Link              |
|-----------|----------------------------------------------|:------------------------------------:|
|ADE20k     |  22k images with 434k instances (total)      |  [official site][ADE20k]             |
|OpenImages |  944k images with 2.6M instances (total)     |  [official site][OpenImages]         |
|MS COCO    |  118k images with 1.2M instances (train)     |  [official site][MSCOCO]             |
|LVIS v1.0  |  100k images with 1.2M instances (total)     |  [official site][LVIS]               |
|COCO+LVIS* |  99k images with 1.5M instances (train)      |  [original LVIS images][LVIS] + <br> [our combined annotations][COCOLVIS_annotation] |
|SBD        |  8498 images with 20172 instances for (train)<br>2857 images with 6671 instances for (test) |[official site][SBD]|
|Grab Cut   |  50 images with one object each (test)       |  [GrabCut.zip (11 MB)][GrabCut]      |
|Berkeley   |  96 images with 100 instances (test)         |  [Berkeley.zip (7 MB)][Berkeley]     |
|DAVIS      |  345 images with one object each (test)      |  [DAVIS.zip (43 MB)][DAVIS]          |
|Pascal VOC |  1449 images with 3417 instances (validation)|  [official site][PascalVOC]          |
|COCO_MVal  |  800 images with 800 instances (test)        |  [COCO_MVal.zip (127 MB)][COCO_MVal] |

[ADE20k]: http://sceneparsing.csail.mit.edu/
[OpenImages]: https://storage.googleapis.com/openimages/web/download.html
[MSCOCO]: https://cocodataset.org/#download
[LVIS]: https://www.lvisdataset.org/dataset
[SBD]: http://home.bharathh.info/pubs/codes/SBD/download.html
[GrabCut]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/GrabCut.zip
[Berkeley]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/Berkeley.zip
[DAVIS]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/DAVIS.zip
[PascalVOC]: http://host.robots.ox.ac.uk/pascal/VOC/
[COCOLVIS_annotation]: https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/cocolvis_annotation.tar.gz
[COCO_MVal]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/COCO_MVal.zip

Don't forget to change the paths to the datasets in [config.yml](config.yml) after downloading and unpacking.

(*) To prepare COCO+LVIS, you need to download original LVIS v1.0, then download and unpack our 
pre-processed annotations that are obtained by combining COCO and LVIS dataset into the folder with LVIS v1.0.

## Testing

### Pretrained models
We provide pretrained models with different backbones for interactive segmentation.

You can find model weights and evaluation results in the tables below:

<table>
    <thead align="center">
        <tr>
            <th rowspan="2"><span style="font-weight:bold">Train</span><br><span style="font-weight:bold">Dataset</span></th>
            <th rowspan="2">Model</th>
            <th colspan="2">GrabCut</th>
            <th>Berkeley</th>
            <th colspan="2">SBD</th>    
            <th colspan="2">DAVIS</th>
            <th>Pascal<br>VOC</th>
            <th>COCO<br>MVal</th>
        </tr>
        <tr>
            <td>NoC<br>85%</td>
            <td>NoC<br>90%</td>
            <td>NoC<br>90%</td>
            <td>NoC<br>85%</td>
            <td>NoC<br>90%</td>
            <td>NoC<br>85%</td>
            <td>NoC<br>90%</td>
            <td>NoC<br>85%</td>
            <td>NoC<br>90%</td>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td rowspan="1">SBD</td>
            <td align="left"><a href="https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/sbd_h18_itermask.pth">HRNet18 IT-M<br>(38.8 MB)</a></td>
            <td>1.76</td>
            <td>2.04</td>
            <td>3.22</td>
            <td><b>3.39</b></td>
            <td><b>5.43</b></td>
            <td>4.94</td>
            <td>6.71</td>
            <td><ins>2.51</ins></td>
            <td>4.39</td>
        </tr>
        <tr>
            <td rowspan="4">COCO+<br>LVIS</td>
            <td align="left"><a href="https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/coco_lvis_h18_baseline.pth">HRNet18<br>(38.8 MB)</a></td>
            <td>1.54</td>
            <td>1.70</td>
            <td>2.48</td>
            <td>4.26</td>
            <td>6.86</td>
            <td>4.79</td>
            <td>6.00</td>
            <td>2.59</td>
            <td>3.58</td>
        </tr>
        <tr>
            <td align="left"><a href="https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/coco_lvis_h18s_itermask.pth">HRNet18s IT-M<br>(16.5 MB)</a></td>
            <td>1.54</td>
            <td>1.68</td>
            <td>2.60</td>
            <td>4.04</td>
            <td>6.48</td>
            <td>4.70</td>
            <td>5.98</td>
            <td>2.57</td>
            <td>3.33</td>
        </tr>
        <tr>
            <td align="left"><a href="https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/coco_lvis_h18_itermask.pth">HRNet18 IT-M<br>(38.8 MB)</a></td>
            <td><b>1.42</b></td>
            <td><b>1.54</b></td>
            <td><ins>2.26</ins></td>
            <td>3.80</td>
            <td>6.06</td>
            <td><ins>4.36</ins></td>
            <td><ins>5.74</ins></td>
            <td><b>2.28</b></td>
            <td><ins>2.98</ins></td>
        </tr>
        <tr>
            <td align="left"><a href="https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/coco_lvis_h32_itermask.pth">HRNet32 IT-M<br>(119 MB)</a></td>
            <td><ins>1.46</ins></td>
            <td><ins>1.56</ins></td>
            <td><b>2.10</b></td>
            <td><ins>3.59</ins></td>
            <td><ins>5.71</ins></td>
            <td><b>4.11</b></td>
            <td><b>5.34</b></td>
            <td>2.57</td>
            <td><b>2.97</b></td>
        </tr>
    </tbody>
</table>


### Evaluation

We provide the script to test all the presented models in all possible configurations on GrabCut, Berkeley, DAVIS, 
Pascal VOC, and SBD. To test a model, you should download its weights and put them in `./weights` folder (you can 
change this path in the [config.yml](config.yml), see `INTERACTIVE_MODELS_PATH` variable). To test any of our models, 
just specify the path to the corresponding checkpoint. Our scripts automatically detect the architecture of the loaded model.

The following command runs the NoC evaluation on all test datasets (other options are displayed using '-h'):

```.bash
python3 scripts/evaluate_model.py <brs-mode> --checkpoint=<checkpoint-name>
```

Examples of the script usage:
```.bash
# This command evaluates HRNetV2-W18-C+OCR ITER-M model in NoBRS mode on all Datasets.
python3 scripts/evaluate_model.py NoBRS --checkpoint=hrnet18_cocolvis_itermask_3p

# This command evaluates HRNet-W18-C-Small-v2+OCR ITER-M model in f-BRS-B mode on all Datasets.
python3 scripts/evaluate_model.py f-BRS-B --checkpoint=hrnet18s_cocolvis_itermask_3p

# This command evaluates HRNetV2-W18-C+OCR ITER-M model in NoBRS mode on GrabCut and Berkeley datasets.
python3 scripts/evaluate_model.py NoBRS --checkpoint=hrnet18_cocolvis_itermask_3p --datasets=GrabCut,Berkeley
```

### Jupyter notebook

You can also interactively experiment with our models using [test_any_model.ipynb](./notebooks/test_any_model.ipynb) Jupyter notebook.

## Training

We provide the scripts for training our models on the SBD dataset. You can start training with the following commands:
```.bash
# ResNet-34 non-iterative baseline model
python3 train.py models/noniterative_baselines/r34_dh128_cocolvis.py --gpus=0 --workers=4 --exp-name=first-try

# HRNet-W18-C-Small-v2+OCR ITER-M model
python3 train.py models/iter_mask/hrnet18s_cocolvis_itermask_3p.py --gpus=0 --workers=4 --exp-name=first-try

# HRNetV2-W18-C+OCR ITER-M model
python3 train.py models/iter_mask/hrnet18_cocolvis_itermask_3p.py --gpus=0,1 --workers=6 --exp-name=first-try

# HRNetV2-W32-C+OCR ITER-M model
python3 train.py models/iter_mask/hrnet32_cocolvis_itermask_3p.py --gpus=0,1,2,3 --workers=12 --exp-name=first-try
```

For each experiment, a separate folder is created in the `./experiments` with Tensorboard logs, text logs, 
visualization and checkpoints. You can specify another path in the [config.yml](config.yml) (see `EXPS_PATH` 
variable).

Please note that we trained ResNet-34 and HRNet-18s on 1 GPU, HRNet-18 on 2 GPUs, HRNet-32 on 4 GPUs 
(we used Nvidia Tesla P40 for training). To train on a different GPU you should adjust the batch size using the command
line argument `--batch-size` or change the default value in a model script.

We used the pre-trained HRNetV2 models from [the official repository](https://github.com/HRNet/HRNet-Image-Classification). 
If you want to train interactive segmentation with these models, you need to download the weights and specify the paths to 
them in [config.yml](config.yml).

## License

The code is released under the MIT License. It is a short, permissive software license. Basically, you can do whatever you want as long as you include the original copyright and license notice in any copy of the software/source. 
## Citation

If you find this work is useful for your research, please cite our papers:
```
@article{reviving2021,
  title={Reviving Iterative Training with Mask Guidance for Interactive Segmentation},
  author={Sofiiuk, Konstantin and Petrov, Ilia and Konushin, Anton},
  journal={arXiv preprint arXiv:2102.06583},
  year={2021}
}

@inproceedings{fbrs2020,
   title={f-brs: Rethinking backpropagating refinement for interactive segmentation},
   author={Sofiiuk, Konstantin and Petrov, Ilia and Barinova, Olga and Konushin, Anton},
   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
   pages={8623--8632},
   year={2020}
}
```
