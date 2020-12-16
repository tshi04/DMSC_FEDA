# DMSC_FEDA
Document Level Multi-Aspect Rating Prediction

[![image](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![image](https://img.shields.io/pypi/l/ansicolortags.svg)](https://github.com/tshi04/DMSC_FEDA/blob/master/LICENSE)
[![image](https://img.shields.io/github/contributors/Naereen/StrapDown.js.svg)](https://github.com/tshi04/DMSC_FEDA/graphs/contributors)
[![image](https://img.shields.io/github/issues/Naereen/StrapDown.js.svg)](https://github.com/tshi04/DMSC_FEDA/issues)
[![image](https://img.shields.io/badge/arXiv-1805.09461-red.svg?style=flat)](https://arxiv.org/pdf/2009.09112.pdf)

This repository is a pytorch implementation for the following arxiv paper:

#### [Deliberate Self-Attention Network with Uncertainty Estimation for Multi-Aspect Review Rating Prediction](https://arxiv.org/pdf/2009.09112.pdf)
[Tian Shi](http://people.cs.vt.edu/tshi/homepage/home), 
[Ping Wang](http://people.cs.vt.edu/ping/homepage/), 
[Chandan K. Reddy](http://people.cs.vt.edu/~reddy/)

## Requirements

- Python 3.6.9
- argparse=1.1
- torch=1.4.0
- sklearn=0.22.2.post1
- numpy=1.18.2

## Dataset

Some data sources:

- TripAdvisor-R:
[https://github.com/HKUST-KnowComp/DMSC](https://github.com/HKUST-KnowComp/DMSC)
- TripAdvisor-RU:
[https://github.com/Junjieli0704/HUARN](https://github.com/Junjieli0704/HUARN)
- TripAdvisor-B:
[https://github.com/HKUST-KnowComp/VWS-DMSC](https://github.com/HKUST-KnowComp/VWS-DMSC)
- BeerAdvocate-R:
[https://github.com/HKUST-KnowComp/DMSC](https://github.com/HKUST-KnowComp/DMSC)
- BeerAdvocate-B:
[https://github.com/HKUST-KnowComp/VWS-DMSC](https://github.com/HKUST-KnowComp/VWS-DMSC)

Please download processed dataset from here. Place them along side with DMSC_FEDA.

```bash
|--- DMSC_FEDA
|--- Data
|    |--- trip_rate
|    |--- trip_binary
|    |    |--- dev
|    |    |--- dev.bert
|    |    |--- glove_42B_300d.npy
|    |    |--- test
|    |    |--- test.bert
|    |    |--- train
|    |    |--- train.bert
|    |    |--- vocab
|    |    |--- vocab_glove_42B_300d
|--- nats_results (results, automatically build)
```

## Usuage

```Training, Validate, Testing``` python3 run.py --task train
```Testing only``` python3 run.py --task test
```Evaluation``` python3 run.py --task evaluate
```keywords Extraction``` python3 run.py --task keywords
```Attention Weight Visualization``` python3 run.py --task visualization

If you want to run baselines, you may need un-comment the corresponding line in ```run.py```.

## Baselines Implemented.

| Model | BRIEF | 
| ------ | ------ |
| mtCNN | CNN + Multi-Task Learning |
| mtRNN | Bi-LSTM + Multi-Task Learning |
| mtAttn | mtRNN + Self-Attention |
| mtBertAttn | BERT + Multi-Task Learning + Self-Attention |
| mtAttnDA | mtRNN + Deliberated Self-Attention |
| MtAttnFE | mtAttn + Pretrained Embedding + Feature Enrichment |
| FEDA | mtAttnDA + Pretrained Embedding + Feature Enrichment |

- Glove_42B_300D word embedding.

# Use Pretrained Model

Coming Soon.

## Citation

```
@article{shi2020deliberate,
  title={Deliberate Self-Attention Network with Uncertainty Estimation for Multi-Aspect Review Rating Prediction},
  author={Shi, Tian and Wang, Ping and Reddy, Chandan K},
  journal={arXiv preprint arXiv:2009.09112},
  year={2020}
}
```
