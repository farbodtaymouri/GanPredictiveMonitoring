# GanPredictiveMonitoring
Predict last event in a business process.

To cite this code use the following:

@InProceedings{10.1007/978-3-030-58666-9_14,
author="Taymouri, Farbod
and Rosa, Marcello La
and Erfani, Sarah
and Bozorgi, Zahra Dasht
and Verenich, Ilya",
editor="Fahland, Dirk
and Ghidini, Chiara
and Becker, J{\"o}rg
and Dumas, Marlon",
title="Predictive Business Process Monitoring via Generative Adversarial Nets: The Case of Next Event Prediction",
booktitle="Business Process Management",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="237--256",
abstract="Predictive process monitoring aims to predict future characteristics of an ongoing process case, such as case outcome or remaining timestamp. Recently, several predictive process monitoring methods based on deep learning such as Long Short-Term Memory or Convolutional Neural Network have been proposed to address the problem of next event prediction. However, due to insufficient training data or sub-optimal network configuration and architecture, these approaches do not generalize well the problem at hand. This paper proposes a novel adversarial training framework to address this shortcoming, based on an adaptation of Generative Adversarial Networks (GANs) to the realm of sequential temporal data. The training works by putting one neural network against the other in a two-player game (hence the ``adversarial'' nature) which leads to predictions that are indistinguishable from the ground truth. We formally show that the worst-case accuracy of the proposed approach is at least equal to the accuracy achieved in non-adversarial settings. From the experimental evaluation it emerges that the approach systematically outperforms all baselines both in terms of accuracy and earliness of the prediction, despite using a simple network architecture and a naive feature encoding. Moreover, the approach is more robust, as its accuracy is not affected by fluctuations over the case length.",
isbn="978-3-030-58666-9"
}

Note: To get faster results, it is recommended to run the codes on GPUs. Note that, after the training is done, your results might be a slightly different than thoes
reported in the paper, and it can be attributed to the min-max game optimization. Indeed, its convergence varies from one experiment to the other one, 
as it is stated in the paper. 
