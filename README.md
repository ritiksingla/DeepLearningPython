# Deep Learning From Scratch
Highly inspired from [DLFS_code](https://github.com/SethHWeidman/DLFS_code)

## Run Locally
To run locally make changes in test.py file

## Currently supports

1. **Layers** : Dense, BatchNormalization, Flatten, Conv2D, MaxPool2D
2. **Optimizers**: SGD with optional momentum, RMSProp, Adam
3. **Activation Functions** : Linear, Sigmoid, Tanh, ReLU
4. **Losses** : MSELoss, L1Loss, SoftmaxCrossEntropy
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
	Dataset: sklearn digits dataset with 1797 examples of 8x8x1 images,
	training dataset : 1257 examples, test dataset : 540 examples

	1. Single Hidden Layer
	RMSProp(lr = 0.01), loss = BCELoss
	Training Accuracy : 99%
	Test Accuracy : 95%
```