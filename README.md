# Plant Disease Classification using CNN

## Introduction

This project demonstrates the application of Convolutional Neural Networks (CNN) for classifying plant diseases using images. The model is trained on the PlantVillage dataset, which contains images of healthy and diseased plant leaves. The CNN model is designed to identify the type of disease based on the visual symptoms present on the leaves.

## Dataset

The dataset used in this project is the PlantVillage dataset. It contains images of plant leaves categorized into different classes based on the type of disease. The dataset is divided into training, validation, and test sets.
https://www.kaggle.com/datasets/faysalmiah1721758/potato-dataset

## Data Preprocessing

### Loading and Visualizing Data

The dataset is loaded using TensorFlow's `image_dataset_from_directory` function. The images are resized to 256x256 pixels, and batches of 32 images are created for training. The class names are extracted from the dataset directory structure.

### Splitting the Dataset

The dataset is split into training, validation, and test sets. The training set comprises 80% of the data, while the validation and test sets each comprise 10% of the data. This ensures a balanced distribution of samples for model evaluation.

### Data Augmentation

To improve the model's robustness and performance, data augmentation techniques such as random flipping and rotation are applied to the training dataset. This helps in creating variations of the images and prevents overfitting.

## Model Building

### Model Architecture

The model architecture is built using a sequential CNN. It includes the following layers:
- **Resizing and Rescaling Layer**: Resizes images to the desired size and normalizes pixel values.
- **Data Augmentation Layer**: Applies random transformations to the images.
- **Convolutional Layers**: Extracts features from the images using convolution operations.
- **MaxPooling Layers**: Reduces the spatial dimensions of the feature maps.
- **Flatten Layer**: Converts the 2D feature maps into a 1D vector.
- **Dense Layers**: Performs classification using fully connected layers with ReLU and Softmax activations.

### Compiling the Model

The model is compiled using the Adam optimizer, Sparse Categorical Crossentropy loss function, and accuracy as the evaluation metric.

### Training the Model

The model is trained for 10 epochs using the training dataset. The validation dataset is used to monitor the model's performance and prevent overfitting.

## Model Evaluation

The trained model is evaluated on the test dataset to measure its accuracy. The accuracy and loss curves are plotted to visualize the training and validation performance over the epochs.

### Inference

A function is created to perform inference on new images. The function predicts the class of the image and calculates the confidence score of the prediction. Sample images from the test dataset are used to demonstrate the inference process.

## Saving the Model

The trained model is saved for future use. Each new version of the model is appended to the list of models as a new version.

## Conclusion

This README provides an overview of the process of building, training, and evaluating a CNN model for plant disease classification. The model achieves high accuracy and can be used for automated plant disease detection, which can help in early diagnosis and treatment of plant diseases.
