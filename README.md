# Deep Learning From Scratch

## Run Locally
To run locally make changes in test.py file

## Currently supports

1. **Layers** : Dense, BatchNormalization, Flatten, Conv2D
2. **Optimizers**: SGD with optional momentum, RMSProp, Adam
3. **Activation Functions** : Linear, Sigmoid, Tanh, ReLU, Softmax
4. **Losses** : MSELoss, L1Loss, BCELoss
5. **LR Schedulers** : ReduceLROnPlateau, ExponentialLR

## Results


### Neural Networks
1. Linear Regression
```
	Dataset:
	Single continous output with 13 features and 506 examples,
	training dataset : 354 examples, test dataset : 152 examples
	
	Metrics:
	1. No hidden Layer
	SGD(lr = 0.01), loss = MSELoss
	Training Accuracy : 74%
	Test Accuracy : 72%

	2. Single hidden Layer
	Adam(lr = 0.1), loss = MSELoss
	Training Accuracy : 85%
	Test Accuracy : 80%

	3. Two hidden Layers with Batch Norm
	Adam(lr = 0.01, nesterov = True), loss = MSELoss
	Training Accuracy : 88%
	Test Accuracy : 85%
```
2. Classification
```
	Dataset: 4 classes output with 500 examples and 25 features,
	training dataset : 350 examples, test dataset : 150 examples

	1. Single Hidden Layer
	RMSProp(lr = 0.01), loss = BCELoss
	Training Accuracy : 97%
	Test Accuracy : 79%
```

### Convolutional Networks
- Classification
```
	Dataset: MNIST dataset with 2000 examples,
	training dataset : 1400 examples, test dataset : 600 examples

	1. Single Hidden Layer
	RMSProp(lr = 0.01), loss = BCELoss
	Training Accuracy : 80%
	Test Accuracy : 80%
```