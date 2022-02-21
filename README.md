# GETM: Graph Embedded Topic Model
A generative model that intergrates embedded topic model and node2vec.

The detailed description and its application on UK-Biobank could be found [here](https://www.biorxiv.org/content/10.1101/2022.01.07.475444v1)

## Contents ##
- [Contents](#contents)
	- [1 Model Overview](#1-model-overview)
	- [2 Code Organization](#2-code-organization)
	- [3 Running Example](#3-running-example)
	- [4 References](#4-references)

## 1 Model Overview
![](doc/methods.png)(#getm model overview and its application on multi-type medical features)
**(a)** GETM training. GETM is a variational autoencoder (VAE) model. The neural network encoder takes individuals' condition and medication information as input and produces the variational mean μ and variance σ<sup>2</sup>> for the patient topic mixtures **θ**. The decoder is linear and consists of two tri-factorizations. One learns medication-defined topic embedding **α**<sup>(med)</sup> and medication embedding **ρ**<sup>(med)</sup>. The other learns condition-specific topic embedding **α**<sup>(cond)</sup> and the condition embedding **ρ**<sup>(cond)</sup>. We separately pre-train **(b)** the embedding of medications **ρ**<sup>(med)</sup> and **(c)** the embedding of conditions **ρ**<sup>(cond)</sup> using node2vec based on their structural meta-information. This is done learning the node embedding that maximizes the likelihood of the tree-structured relational graphs of conditions and medications.

## 2 Code Organization

## 3 Running Example

## 4 References
<a id="1">[1]</a>
Aditya Grover and Jure Leskovec. node2vec: Scalable feature learning for networks.
CoRR, abs/1607.00653, 2016.





