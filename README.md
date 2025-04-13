# CNN-Architecture
Overview
This notebook demonstrates the implementation of Convolutional Neural Networks (CNNs) to classify handwritten digits from the MNIST dataset. It includes two architectures: LeNet-5 and AlexNet, showcasing their training, evaluation, and performance visualization.

Features
1. LeNet-5 Architecture
Model Definition:

Input layer for 28x28 grayscale images.

Two convolutional layers with average pooling.

Fully connected layers for classification.

Softmax activation for multiclass classification.

Compilation:

Loss function: Categorical Crossentropy.

Optimizer: Adam.

Metric: Accuracy.

Training:

Trains the model on MNIST data for 20 epochs with a batch size of 32.

Performance Visualization:

Plots training and validation accuracy and loss over epochs.

2. AlexNet Architecture
Model Definition:

Input resized to 227x227 pixels.

Multiple convolutional layers with max pooling.

Fully connected layers with dropout regularization.

Preprocessing:

Rescales pixel values to.

Resizes images to match AlexNet's input requirements.

3. Data Preprocessing
Normalizes pixel values to.

Adds a channel dimension to grayscale images (28x28x1).

One-hot encodes labels for multiclass classification.

Dependencies
The following Python libraries are required:

tensorflow: For building and training CNN models.

numpy: For numerical computations.

matplotlib: For visualizing training metrics.

Installation
To install the required libraries, run:

bash
pip install tensorflow numpy matplotlib
Usage Instructions
Clone or download the notebook file to your local machine.

Open the notebook in Jupyter Notebook or Google Colab.

Run all cells sequentially to execute the analysis.

Key Sections
LeNet-5 Implementation
Model Architecture:

python
model = Sequential([
    InputLayer(input_shape=(28,28,1)),
    Conv2D(16, kernel_size=(5,5), activation='sigmoid'),
    AveragePooling2D(pool_size=(2,2)),
    Conv2D(32, kernel_size=(5,5), activation='sigmoid'),
    AveragePooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(120, activation='sigmoid'),
    Dense(84, activation='sigmoid'),
    Dense(10, activation='softmax')
])
Training:

python
model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))
Performance Visualization:

python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.show()
AlexNet Implementation
Input Resizing:

python
x_train = tf.image.resize(x_train, (227,227))
x_test = tf.image.resize(x_test, (227,227))
Model Architecture:
AlexNet uses multiple convolutional layers followed by max pooling and dropout regularization.

Observations
LeNet-5 Results:

Training accuracy reaches ~99%.

Validation accuracy stabilizes around ~98%.

AlexNet Results:

Suitable for larger input sizes and complex datasets.

The decreasing validation loss indicates good generalization performance.

Future Improvements
Implement early stopping to prevent overfitting.

Experiment with advanced architectures like ResNet or VGG for larger datasets.

Use data augmentation techniques to improve model robustness.

License
This project is open-source and available under the MIT License.
