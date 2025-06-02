# 🧠 LeNet and Basics of Neural Networks

Welcome to this repository dedicated to understanding the fundamentals of neural networks, with a hands-on implementation of the **LeNet-5** architecture.

## 📁 Contents

- `LeNet.ipynb` – A Jupyter Notebook demonstrating the LeNet-5 architecture using a sample dataset (e.g., MNIST).
- `dataset/` – Directory containing the dataset used to train and evaluate the model.
- `document/` – A brief document explaining the core concepts of neural networks and the architecture of LeNet.

---

## 📌 Project Overview

This repository serves as an educational resource for:

- Understanding the building blocks of a neural network (perceptrons, activation functions, etc.)
- Learning how convolutional neural networks (CNNs) operate
- Implementing the classic LeNet-5 model in Python (likely using TensorFlow or PyTorch)
- Training and evaluating the model on a small image dataset

---

## 🧾 LeNet-5: At a Glance

**LeNet-5** is one of the earliest CNN architectures developed by Yann LeCun. It was designed primarily for digit recognition and forms the basis for many modern CNNs.

Key layers:
- Convolutional Layers
- Subsampling (Pooling) Layers
- Fully Connected Layers

> LeNet was originally applied to the MNIST dataset for handwritten digit classification.

---

## 🚀 Getting Started

### Requirements

Make sure you have the following Python packages installed:

```bash
pip install numpy matplotlib tensorflow notebook
```

Or if using PyTorch:
```
pip install torch torchvision
```
### Running the Notebook
```
jupyter notebook LeNet-5.ipynb
```
## 📊 Dataset

The dataset used is stored in the `dataset/` folder. This could be:

- A subset of MNIST  
- Custom handwritten digits or small image classification dataset

Make sure the paths inside the notebook match the dataset location.

---

## 📄 Document

The `document/` folder contains a write-up or PDF/Markdown explaining:

- What neural networks are  
- How they function (with diagrams)  
- Overview of LeNet-5's architecture  
- Training steps and evaluation

---

## 📈 Results

The notebook contains visualizations of:

- Training/validation accuracy and loss  
- Sample predictions  
- Model architecture summary

---

