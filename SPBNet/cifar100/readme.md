
# 1-D SPATIAL ATTENTION IN BINARIZED CONVOLUTIONAL NEURAL NETWORKS

- For running this evaluation, you should have CIFAR-100 data in **./data** directory. 
- Before you run the training, **./results** directory should be pre-made. -- In **./results**, a directory is automatically generated, where training checkpoints and logging data are stored. 
- This baseline code is borrowed from ReActNet18 (from ResNet18). 
- The binarized convolution module is written in dorefanet.py, which supports n-bit quantization. But n-bit quantized convolutions were not used. 

- SpatialSE module written in net.py has parameters for the reduction ratios. 

- In sp4-lb, you can see averaged accuracy of 73.5%@Top-1.

- Data (Best validation accuracy) - all data are collected from the training from scratch. We adopted 5 different seeds (42-46). Attached \*.out files contain the training data and accuracy per each epoch.










