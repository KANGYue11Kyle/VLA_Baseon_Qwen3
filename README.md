# VLA_Baseon_Qwen3
# Laparoscopic Prediction Based on the QwenVLA Model

A training and evaluation project for a surgical video understanding model based on QwenVL, supporting multi-frame sampling, dataset preprocessing, per-round checkpoint evaluation, and visualization.

## 📁 Project Structure

```
.
├── checkpoint-1600/                          # Best model weights (corresponding to the 1600th training step)
├── qwenvl/                                   # Core training code and dataset splitting logic
├── 12frames_6sample_pre.py                   # Dataset preprocessing and splitting (12 images / 6 samples format)
├── evaluate_checkpoints.py                   # Per-round validation of model accuracy for each checkpoint
├── model_performance_summary.png             # Model performance summary figure (reserved placeholder)
├── surgery_train_12frames_6sample...         # Training set data split and display file
├── surgery_val_12frames_6samples.j...        # Validation set data split and display file
└── vis.py                                    # Script for generating the visualization results list
```

## 🔍 Module Description

| File / Directory | Description |
| --- | --- |
| `checkpoint-1600/` | The best model weights selected from this training run, corresponding to the 1600th training step |
| `qwenvl/` | Stores the core training code and dataset splitting logic |
| `12frames_6sample_pre.py` | Performs dataset preprocessing and splitting in the 12-image, 6-sample format |
| `evaluate_checkpoints.py` | Per-round validation of model accuracy for each checkpoint |
| `model_performance_summary.png` | Reserved placeholder for the model performance summary figure |
| `surgery_train_12frames_6sample...` | Training set data split and display file |
| `surgery_val_12frames_6samples.j...` | Validation set data split and display file |
| `vis.py` | Script for generating the visualization results list |

## 🚀 Quick Start

### Environment Dependencies

> TODO: Add the Python version, CUDA version, and dependency installation commands

```bash
# Example
pip install -r requirements.txt
```

### Dataset Preprocessing

Split the dataset in the 12-image, 6-sample format:

```bash
python 12frames_6sample_pre.py
```

### Model Training

> TODO: Add the training launch command and related parameter descriptions

```bash
# Example
bash qwenvl/train.sh
```

### Model Evaluation

Validate the accuracy of each checkpoint round by round:

```bash
python evaluate_checkpoints.py
```

### Result Visualization

Generate the visualization results list:

```bash
python vis.py
```

## 📊 Model Performance

> TODO: After training is complete, place the performance summary figure at `model_performance_summary.png`

![Model Performance Summary](model_performance_summary.png)

## 🏆 Best Weights

The best model weights selected from this training run are **`checkpoint-1600`**, corresponding to the **1600**th training step.


## 🙏 Acknowledgements

The Qwen3 open-source project
