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

### 3.3.

---

## 4. References

1. **DLinear(Zeng et al., 2022)** model implementation from : [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)

---

## Author

-   Woojin Choi <cwwojin@gmail.com> <br/>
