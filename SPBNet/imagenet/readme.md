

# 1-D SPATIAL ATTENTION IN BINARIZED CONVOLUTIONAL NEURAL NETWORKS

- For running this evaluation, you should have ImageNet data in **./data** directory. 
- Before you run the training, **./results** directory should be pre-made. -- In **./results**, a directory is automatically generated, where training checkpoints and logging data are stored. 
- This baseline code is borrowed from ReActNet-A (from MobileNet). 
- The binarized convolution module is written in dorefanet.py, which supprots n-bit quantization. But n-bit quantized convolutions were not used. 
- To finish total training on our maching having 80 threads CPU, 6 A5000 GPUs, 20 days are required without ReActNet-A triangle derivative of Sign function. If the triangle derivative is used, about 30% additional training time will be needed in our evaluations.
 
- SpatialSE module written in xmobilenet.py has parameters for the reduction ratios. 

- In SPBNet-A with sp4-lb, you can see the best accuracy of 70.7%@Top-1. Attached \*.out files contain the training data and accuracy per each epoch.

- Data (Best validation accuracy)
imagenet_scratch.out: activation: 1bit, weights: 1 bits trained from scratch (69.4%@Top-1)

imagenet_ts_first.out: activation: 1bit, weights: 32 bits of the first training step using teacher-student training (71.9%@Top-1)
imagenet_ts_second.out: activation: 1bit, weights: 1 bits of the second training step using teacher-student training (70.7%@Top-1)







