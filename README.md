# Introduction 📖
This project, titled "Deep Learning Optimization for Tomato Plant Disease Image Classification," is an academic endeavor assigned by Asia Pacific University Malaysia. The primary goal of this project is to implement and optimize a deep learning model to accurately classify diseases in tomato plants from images.

In this project, I will be working with a comprehensive dataset of tomato plant images, each annotated with specific disease labels. The focus will be on optimizing the deep learning model to achieve the best possible classification accuracy. The chosen model for this task is a hybrid **ResNet50-BiLSTM** model, which combines the strengths of ResNet50's deep convolutional neural network capabilities with BiLSTM's sequential data processing efficiency.

The ResNet50 component will act as a feature extractor, identifying and extracting relevant features from the tomato plant images. These features will then be processed by the BiLSTM component to capture temporal dependencies and patterns, further enhancing the model's ability to classify diseases accurately.

Throughout the implementation, various optimization techniques will be explored to improve the model's performance. These techniques may include hyperparameter tuning, data augmentation, transfer learning, and fine-tuning of the model architecture.

# Aim & Objectives 🎯
**Aim**: The aim of the project is to develop and optimize a deep learning model that can accurately classify different diseases in tomato plants from images.

**Objectives**:
* To retrieve a comprehensive image dataset related to tomato plant diseases from the Kaggle website.
* To implement and evaluate a deep learning model for tomato plant disease classification.
* To optimize the deep learning model based on the evaluation results of different model variants.

# Dataset 🛢️
* **Dataset**: (New Plant Diseases Dataset)[https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset]
* **Description**: This dataset comes with three subsets, known as training, validation, and test. It consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes. However, the classes related to tomatoes have been chosen for the success of this project. Therefore, there are a total of 10 classes focused on in this project: one healthy tomato plant class and nine tomato plant disease classes. The following are the 10 classes: **Bacterial Spot**, **Early Blight**, **Healthy**, **Late Blight**, **Leaf Mold**, **Septoria Leaf Spot**, **Spider Mites**, **Target Spot**, **Mosaic Virus**, **Yellow Leaf Curl Virus**.
