# LHCompas_2024

-   [LHCompas_2024](#lhcompas_2024)
    -   [1. Introduction](#1-introduction)
    -   [2. Getting Started](#2-getting-started)
    -   [3. Contribution](#3-contribution)
    -   [4. References](#4-references)
    -   [Author](#author)

## 1. Introduction

This repository contains trainers for models for : **Time-Series Forecasting for Commercial Vacancy-Rates**

## 2. Getting Started

This is the installation guide.

### 2.1. Install with pip

```shell
$ pip install -e compas/
```

Testing done using conda environment.

### 2.2. Run an Experiment

```shell
$ python compas/experiments/run_LSTM.py --config="compas/experiments/config/lstm.yaml"
```

To run your own experiments, edit the `.yaml` file and pass as argument.

### 2.3. Run Inference with trained Model

```python
import pandas as pd
from compas.inference import ForecastModel

model = ForecastModel(model_path="path/to/model")
df = pd.read_csv("path/to/data.csv")

model.forecast(df, steps=12)
```

## 3. Contribution

### 3.1. Branch Naming

-   feature/{$feature_branch_name}
-   hotfix/{$hotfix_branch_name}
-   release/{$release_ver}

### 3.2. Using Templates

-   Run this to set the commit message template.

```shell
$ git config commit.template .gitmessage.txt
```

### 3.3. Code Formatting with Black

-   We use the `Black` formatter with its default settings for this repository.

```shell
$ pip install black
$ black compas/
```

---

## 4. References

1. **DLinear(Zeng et al., 2022)** model implementation from : [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)

---

## Author

-   Woojin Choi <cwwojin@gmail.com> <br/>
