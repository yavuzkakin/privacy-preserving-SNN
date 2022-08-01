# Spiking Neural Networks for privacy preserving classification

This repository is meant to host our semester project based on implementing a Spiking Neural Network for classification, and using Crypten [1] to perform testing using privacy-preserving MPC. 

##Collaborators with access to this repository

Nicolas SERVOT : EURECOM Student  
Yavuz AKIN : EURECOM Student  
Oubaida Chouchane : Staff SEC, project supervisor   
Melek ONEN : Staff SEC, project responsible  
Massimiliano TODISCO : Staff SEC, project responsible  

## Content of the repository

*_(Il faut mettre les différents fichiers qu'on a ici et expliquer ce à quoi ils correspondent)_*

## Architecture of the SNN 

The architecture of the spiking Neural Network is inspired from the Tutorials of Friedemann Zenke on Surrogate Gradient Learning [2].

**Modifications to the original architecture :** 

In order to perform secure MPC during the Testing phase, the forward propagation of the SNN model is modified to compute on MPCTensors. The code has been adapted to work with a world size of at least 2 using GPU. 

The Fashion MNIST Dataset has been used for training and testing. That's why in addition to the SNN architecture code, additional spike encoding functions have been added to convert the dataset into sets of spikes.

**Structure of the Neural Network :**  

The SNN has the structure of a Feed Forward NN.  

It is composed of 3 layers of neurons:  
	1. The first layer has 784 neurons  
	2. The hidden has 100 neurons   
	3. The output layer 10 neurons   

The neurons are spiking neurons according to the Leaky-Integrate and Fire model (LIF).

## How to install

Here are the instructions for installing Crypten on Linux, Mac, and AWS : https://crypten.readthedocs.io/en/latest/

Here are the versions of the libraries we used :

_*(ici faut ajouter les version de crypten, pytorch et tout)*_


## How to use 

*_(On peut remplir cette partie une fois qu'on a fixé le nom des fichiers)_*

## What is next ?

* Until now we've worked on the Fashion MNIST Dataset. The next step could be to train and test our model on classification of the spiking Heidelberg digits [3].
* The final step of the project could be the implementation of an SNN for classifying voice spoofing records. 

 

## Ressources

[1] [https://github.com/facebookresearch/crypten](https://github.com/facebookresearch/crypten)   
[2] [https://github.com/fzenke/spytorch](https://github.com/fzenke/spytorch)  
[3][https://compneuro.net/posts/2019-spiking-heidelberg-digits/](https://compneuro.net/posts/2019-spiking-heidelberg-digits/)