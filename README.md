# LHCompas_2024

- [LHCompas\_2024](#lhcompas_2024)
  - [1. Introduction](#1-introduction)
  - [2. Getting Started](#2-getting-started)
  - [3. Run with Docker](#3-run-with-docker)
  - [4. Contribution](#4-contribution)
  - [5. References](#5-references)
  - [Author](#author)

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
$ python compas/experiments/run_Exp.py --config="compas/experiments/config/lstm.yaml"
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

## 3. Run with Docker

### 3.1. Prerequisites

1. Platforms - Tested on
    - MacOS 14.5
    - WSL2 / Ubuntu 22.04
    - Linux - Ubuntu 22.04
2. Install [Docker / Docker Desktop](https://docs.docker.com/desktop/install/linux-install/)

### 3.2. CUDA / GPU Support

To run the app with GPU support, install

1. Required drivers : **CUDA 12.1, with supported Nvidia drivers for Windows / Linux**
2. Install [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
    - Refer to the official documentation for install guides per platform.
    - **Install on Linux w/ APT**
    ```shell
    $ curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    $ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    ```

### 3.3. Launch with Docker Compose

Use the script `launch.sh` to launch all services including the MLFlow Hosting Server.

```shell
$ bash launch.sh
```

To launch specific services / profile, use the Docker Compose CLI

-   `--profile train` : launch the main container only
-   `--profile mlflow` : launch the MLFlow hosting server (3 containers) only

```shell
$ docker compose --profile train up -d
```

## 4. Contribution

### 4.1. Branch Naming

-   feature/{$feature_branch_name}
-   hotfix/{$hotfix_branch_name}
-   release/{$release_ver}

### 4.2. Using Templates

-   Run this to set the commit message template.

```shell
$ git config commit.template .gitmessage.txt
```

### 4.3. Code Formatting with Black

-   We use the `Black` formatter with its default settings for this repository.

```shell
$ pip install black
$ black compas/
```

---

## 5. References

1. **DLinear(Zeng et al., 2022)** model implementation from : [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)

---

## Author

-   Woojin Choi <cwwojin@gmail.com> <br/>
