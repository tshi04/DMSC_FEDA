# DMSC
Document Level Multi-Aspect Sentiment Classification

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

### Experiments and Results

In this section, we show some experimental results on different datasets.

| Model | BRIEF | 
| ------ | ------ |
| MAJOR | Take majority label as predict label |
| MCNN | CNN + Multi-Task Learning |
| MLSTM | Bi-LSTM + Multi-Task Learning |
| MATTN | MLSTM + Self-Attention |
| MBATTN | BERT + Multi-Task Learning + Self-Attention |
| MDSA | MLSTM + Deliberated Self-Attention |
| MATTN-FE | MATTN + Pretrained Embedding + Feature Enrichment |
| MDSA-FE | MDSA + Pretrained Embedding + Feature Enrichment |


- Glove_42B_300D word embedding.

#### TRIPADVISOR-R

|Model|Accuracy-DEV|MSE-DEV|Accuracy-TEST|MSE-TEST|
|-|-|-|-|-|
|MCNN|40.74|1.524|41.75|1.458|
|MLSTM|42.57|1.413|42.74|1.401|
|MATTN|41.94|1.465|42.13|1.427|
|MBATTN|44.14|1.306|44.41|1.250|
|MDSA|43.93|1.357|44.50|1.300|
|MATTN-FE|45.42|1.248|45.70|1.224|
|MDSA-FE|46.05|1.220|46.72|1.178|

#### TRIPADVISOR-RU

|Model|Accuracy-DEV|MSE-DEV|Accuracy-TEST|MSE-TEST|
|-|-|-|-|-|
|MCNN|51.76|0.693|51.21|0.714|
|MLSTM|48.74|0.785|48.64|0.791|
|MATTN|50.73|0.681|50.53|0.678|
|MBATTN|55.12|0.606|54.50|0.617|
|MDSA|54.07|0.625|53.41|0.632|
|MATTN-FE|55.71|0.571|55.39|0.585|
|MDSA-FE|56.11|0.568|55.82|0.574|

#### BEERADVOCATE-R

|Model|Accuracy-DEV|MSE-DEV|Accuracy-TEST|MSE-TEST|
|-|-|-|-|-|
|MCNN|34.30|2.038|34.11|2.016|
|MLSTM|34.81|2.172|34.48|2.167|
|MATTN|36.41|1.941|35.78|1.962|
|MBATTN|36.42|1.958|35.94|1.963|
|MDSA|39.27|1.740|38.92|1.714|
|MATTN-FE|39.76|1.604|38.85|1.633|
|MDSA-FE|39.70|1.624|39.66|1.617|

#### RATEMDS-R

|Model|Accuracy-DEV|MSE-DEV|Accuracy-TEST|MSE-TEST|
|-|-|-|-|-|
|MCNN|45.70|1.330|46.19|1.333|
|MLSTM|47.63|1.172|48.37|1.148|
|MATTN|48.62|1.176|49.08|1.157|
|MBATTN|48.08|1.177|48.47|1.167|
|MDSA|48.62|1.142|49.28|1.123|
|MATTN-FE|49.12|1.126|49.68|1.108|
|MDSA-FE|49.16|1.130|49.80|1.106|

#### TRIPADVISOR-B

|Model|Accuracy-DEV|Accuracy-TEST|
|-|-|-|
|MCNN|81.00|81.31|
|MLSTM|81.12|80.56|
|MATTN|80.51|80.82|
|MBATTN|82.90|82.84|
|MDSA|81.88|82.39|
|MATTN-FE|83.48|83.43|
|MDSA-FE|83.13|84.23|

#### BEERADVOCATE-B

|Model|Accuracy-DEV|Accuracy-TEST|
|-|-|-|
|MCNN|82.21|82.37|
|MLSTM|81.51|82.07|
|MATTN|83.80|84.86|
|MBATTN|83.95|84.73|
|MDSA|84.31|84.99|
|MATTN-FE|85.49|85.99|
|MDSA-FE|85.79|86.52|


#### RATEMDS-B

|Model|Accuracy-DEV|Accuracy-TEST|
|-|-|-|
|MCNN|82.01|81.60|
|MLSTM|82.85|82.40|
|MATTN|83.03|82.66|
|MBATTN|83.61|83.28|
|MDSA|83.22|83.47|
|MATTN-FE|84.06|83.62|
|MDSA-FE|84.08|83.89|




