## Image Classification using PyTorch CNN ##


## Description ##

This is a simple Convolutional Neural Network (CNN) built using **PyTorch** for classifying images.  
The program allows the user to:

1. Train a CNN on a custom dataset.
2. Test the model on a test dataset.
3. Predict the class of a single image interactively.


## Dataset Structure ##

The dataset should follow this structure (example with 2 classes: `lion` and `tiger`):

dataset/
├── train/
│ ├── class1/ # e.g., lion
│ └── class2/ # e.g., tiger
└── test/
├── class1/
└── class2/


Each class folder should contain images of that class.

### Sample Dataset ##

Create a sample dataset with images and add the dataset into the project folder


## Requirements ##

- Python 3.x
- PyTorch
- torchvision
- Pillow (for image loading)
- CUDA (optional, for GPU training)

Install dependencies using pip:

pip install torch torchvision pillow


## How to Run ##

Place your dataset in the project folder.

Run the Project:

python cnn_image_classification.py

Enter the required inputs when prompted:

Dataset path

Number of classes

Batch size

Number of epochs

Learning rate

After training, you can test the model on the test set.

Finally, enter the path of a single image to predict its class.

