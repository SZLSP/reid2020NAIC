## Introduction

This is code for contest [NAIC ReID 2020](https://naic.pcl.ac.cn/contest/6).

After completing the installation, you can write your own config in `configs` folder and run

```bash
python ./tools/train_net.py --gpu-id 0 \
--config-file configs/PATH/TO/YOUR/CONFIG.yml
```

to train your model and run

```bash
python ./tools/submit.py --config-file configs/PATH/TO/YOUR/CONFIG.yml \
  --gpu-id 1 --test-permutation
```

to get the results.

The rest is left for you to explore.

## Installation

See [INSTALL.md](docs/INSTALL.md).

## Quick Start

The designed architecture follows this guide [PyTorch-Project-Template](https://github.com/L1aoXingyu/PyTorch-Project-Template), you can check each folder's purpose by yourself.

See [GETTING_STARTED.md](docs/GETTING_STARTED.md).

Learn more at out [documentation](docs). And see [projects/](projects) for some projects that are build on top of fastreid.

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Fastreid Model Zoo](docs/MODEL_ZOO.md).

## Deployment

We provide some examples and scripts to convert fastreid model to Caffe, ONNX and TensorRT format in [Fastreid deploy](tools/deploy).

## License

Fastreid is released under the [Apache 2.0 license](LICENSE).

## Citing Fastreid

If you use Fastreid in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.

```BibTeX
@article{he2020fastreid,
  title={FastReID: A Pytorch Toolbox for General Instance Re-identification},
  author={He, Lingxiao and Liao, Xingyu and Liu, Wu and Liu, Xinchen and Cheng, Peng and Mei, Tao},
  journal={arXiv preprint arXiv:2006.02631},
  year={2020}
}
```
