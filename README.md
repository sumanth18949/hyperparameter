# Hyperparameter tuning in cnn using gridsearch

This project demonstrates how to build and tune a **Convolutional Neural Network (CNN)** for image classification using the **CIFAR-10 dataset**.  
It includes model training, evaluation, visualization, and hyperparameter optimization using **GridSearchCV**.

---
## 🚀 Project Overview

The goal of this project is to classify 10 types of objects from the CIFAR-10 dataset using a CNN model and improve its performance through **hyperparameter tuning**.

The workflow includes:
- Loading and preprocessing the CIFAR-10 dataset  
- Building a baseline CNN model  
- Training and evaluating the model  
- Visualizing training accuracy and loss  
- Performing hyperparameter tuning using GridSearchCV  
- Comparing tuned vs baseline model performance  

---
## 🧩 Dataset

**CIFAR-10** dataset consists of:
- **60,000 color images** (32×32×3)
- **10 classes:** airplane, bird, cat,frog,ship

The dataset is automatically loaded via:
```python
from tensorflow.keras import datasets
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
