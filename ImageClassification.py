# Import necessary libraries
# OpenCV for image processing
import cv2 as cv
# NumPy for numerical operations and array handling
import numpy as np
# Matplotlib for visualization
import matplotlib.pyplot as plt
# TensorFlow components for building our neural network
from tensorflow.keras import datasets, layers, models

# Load the CIFAR-10 dataset which contains 60,000 32x32 color images in 10 different classes
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1 by dividing by 255 (max RGB value)
training_images, testing_images = training_images/255, testing_images/255

# Define the class names for our 10 categories
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Display a grid of 16 sample images from our training set
for i in range(16):
    # Create a 4x4 grid of subplots, place each image in its position
    plt.subplot(4,4,i+1)
    # Remove tick marks from x and y axes for cleaner visualization
    plt.xticks([])
    plt.yticks([])
    # Display the image in binary colormap
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    # Add the class name as a label below each image
    plt.xlabel(class_names[training_labels[i][0]])

# Show the complete grid of images
plt.show()

# Optional: Reduce dataset size for faster training (commented out)
# training_images = training_images[:20000]
# training_labels = training_labels[:20000]
# testing_images = testing_images[:4000]
# testing_labels = testing_labels[:4000]

# Create a Sequential model - a linear stack of layers
model = models.Sequential()

# First Convolutional Layer
# 32 filters of size 3x3, ReLU activation, input shape matches our 32x32 RGB images
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
# Max Pooling layer to reduce spatial dimensions by half
model.add(layers.MaxPooling2D((2,2)))

# Second Convolutional Layer
# 64 filters of size 3x3, ReLU activation
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

# Third Convolutional Layer
# Another 64 filters of size 3x3
model.add(layers.Conv2D(64, (3,3), activation='relu'))

# Flatten the 3D output to 1D for the dense layers
model.add(layers.Flatten())
# Dense layer with 64 neurons and ReLU activation
model.add(layers.Dense(64, activation='relu'))
# Output layer with 10 neurons (one per class) and softmax activation
model.add(layers.Dense(10, activation='softmax'))

# Configure the training process
# Using Adam optimizer, categorical crossentropy loss (sparse version since our labels are integers)
# Track accuracy during training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model
# Use training images/labels, run for 10 epochs, and validate using test data
model.fit(training_images, training_labels, epochs=10, validation_data = (testing_images, testing_labels))

# Evaluate the model's performance on the test set
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Save the trained model to a file
model.save('image_classifier.keras')

# Load the saved model back (this line appears to be duplicated in the original code)
model = models.load_model('image_classifier.keras')
# Load a test image
img = cv.imread('/content/sample_data/deer.jpg')
# Convert BGR to RGB (OpenCV loads in BGR, but we need RGB)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# Resize the image to match our model's expected input size
img = cv.resize(img, (32, 32))

# Display the image
plt.imshow(img, cmap=plt.cm.binary)
# Make a prediction
# Convert to array, add batch dimension, and normalize pixel values
prediction = model.predict(np.array([img]) / 255)
# Get the index of the highest probability class
index = np.argmax(prediction)
# Print the predicted class name
print(f"Prediction is {class_names[index]}")