# FDTad: A Lightweight Federated Distillation Framework with Dual-channel Transformer for Time-Series Anomaly Detection

This repository contains a partial implementation of FDTad, a lightweight federated distillation framework designed for time-series anomaly detection in distributed environments. The work is part of an ongoing research project.

## Overview

FDTad addresses key challenges in federated learning, including data heterogeneity, privacy constraints, and high communication overhead. The framework combines several innovative components:

1. **Dual-channel Single-head Transformer (DcST)**: A lightweight architecture that captures multi-scale temporal dependencies through feature and temporal channels while maintaining minimal computational overhead.

2. **Adaptive Attention Distillation (AADis)**: A novel knowledge transfer mechanism that dynamically adjusts the intensity of distillation based on model performance, optimizing both accuracy and efficiency.

3. **Joint Optimization Strategy**: Integrates prediction and reconstruction errors to enhance anomaly detection accuracy and robustness.

## Key Features

- **Privacy Preservation**: Federated learning approach that keeps raw data local to devices.
- **Resource Efficiency**: Lightweight design suitable for deployment on resource-constrained devices.
- **Robust Performance**: Effective handling of data heterogeneity and non-IID data distributions.
- **Adaptive Knowledge Transfer**: Dynamic adjustment of distillation intensity based on model performance.

## Repository Structure

This repository contains partial module implementations for the FDTad framework:

- `DcST.py`: Implementation of the Dual-channel Single-head Transformer.
- `AADis.py`: Implementation of the federated distillation mechanism with AADis.

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Additional requirements will be detailed in requirements.txt
