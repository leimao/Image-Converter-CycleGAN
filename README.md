# Image Converter CycleGAN

Lei Mao

University of Chicago

## Introduction

Unlike ordinary pixel-to-pixel translation models, cycle-consistent adversarial networks (CycleGAN) has been proved to be useful for image translations without using paired data. Here is a compact implementation of CycleGAN for image translations.


## Dependencies

* Python 3.5
* Numpy 1.14
* TensorFlow 1.8
* ProgressBar2 3.37.1
* OpenCV 3.4


## Files

```
.
├── data
├── demo
├── download.py
├── LICENSE.md
├── model
├── model.py
├── module.py
├── README.md
├── train.py
└── utils.py
```


## Usage

### Download Dataset

Download and unzip specified dataset to designated directories

```bash
$ python download.py --help
usage: download.py [-h] [--download_dir DOWNLOAD_DIR] [--data_dir DATA_DIR]
                   [--datasets DATASETS [DATASETS ...]]

Download CycleGAN datsets.

optional arguments:
  -h, --help            show this help message and exit
  --download_dir DOWNLOAD_DIR
                        download directory for zipped data
  --data_dir DATA_DIR   data directory for unzipped data
  --datasets DATASETS [DATASETS ...]
                        datasets to download: apple2orange,
                        summer2winter_yosemite, horse2zebra, monet2photo,
                        cezanne2photo, ukiyoe2photo, vangogh2photo, maps,
                        cityscapes, facades, iphone2dslr_flower, ae_photos
```

For example, download ``apple2orange`` and ``horse2zebra`` datasets to ``download`` directory and extract to ``data`` directory:

```bash
$ python download.py --download_dir ./download --data_dir ./data --datasets apple2orange horse2zebra
```

### Train Model

To have a good conversion capability, the training would take at least 100 epochs, which could take very long time even using a NVIDIA GTX TITAN X graphic card. The model also consumes a lot of graphic card memories (> 10 GB). But this could be reduced by reducing the number of convolution filters ``num_filters`` in the model.




### Image Conversion


### Demo


## References

* Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. 2017.
* Xiaowei Hu's CycleGAN TensorFlow Implementation [Repository](https://github.com/xhujoy)
* Hardik Bansal's CycleGAN TensorFlow Implementation [Repository](https://github.com/hardikbansal)

## To-Do List
