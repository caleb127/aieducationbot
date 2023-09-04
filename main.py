import torch.nn as nn
import torch
'''
   - Reshape a tensor from (3, 4) to (4, 3).
3. **Autograd and Gradients:**
   - Define a tensor with requires_grad=True and perform some operations on it.
   - Calculate the gradient of a simple function using autograd.
4. **Linear Regression:**
   - Generate some random data for a simple linear regression problem.
   - Create a linear model in PyTorch, define a loss function, and optimize it using gradient descent.
5. **Neural Network Basics:**
   - Build a small feedforward neural network (FFNN) with one hidden layer.
   - Train the FFNN on a basic dataset like the Iris dataset.
6. **Convolutional Neural Network (CNN):**
   - Implement a simple CNN architecture with convolutional and pooling layers.
   - Train the CNN on a dataset like MNIST for digit classification.
7. **Recurrent Neural Network (RNN):**
   - Create a basic RNN model for sequence prediction.
   - Train it on a toy dataset, like a simple time series.
8. **Save and Load Models:**
   - Save a trained model to a file and then load it to make predictions.
9. **Data Loading:**
   - Learn how to use `torch.utils.data` to load custom datasets and preprocess them for training.
'''
name = torch.zeros(2,3)
print(name)
name2=torch.Tensor([1,2,3,4,5])
print(name2)
name3 = torch.Tensor([1,2,3])
name4=torch.Tensor([4,5,6])
sub=name3-name4
add=name3+name4
print(sub)
print(add)