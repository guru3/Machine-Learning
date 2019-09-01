# List of CNNs that were major breakthroughs of their time!

1. LeNet 5 ( 1998 )
    - Pioneering 7-layered CNN
    - Was used to detect handwritten digits on a cheque
    - Highlighted the importance of CNN than traditional fully connected neural networks
    - Originally used max pooling with tanh activation in hidden layers

2. AlexNet ( 2012 )
    - Defeated all the competitors at ILSVRC, with top-5 error rate of 15%, and the runner up network at 26%
    - Though modelled on LeNet, it was deeper and had more filters per layer
    - It used data augmentation for better accuracy
    - Instead of regularization constant, it used dropouts ( though this doubled the iterations needed to converge! )
    - Also leverage ReLU activations in its hidden layers and used SGD with momentum

3. GoogLeNet ( 2014 )
    - Introduced inception module
    - Achieved a top-5 error rate of 6.67%
    - It used RMSprop for convergence
    - It was 22 layer deep CNN, but used smaller convolutions to significantly reduce total number of parameters
    
4. VGGNet ( 2014 )
    - Significant because of its uniform architecture
    - VGGNet consists of 16 convolutional layers

5. ResNet ( 2015 )
    - Introduced “skip connections”
    - Very deep : 152 layers network
    - Beats human-level performance on IMAGENET dataset

What are ways to get better performance on a neural networks ? We can try changing neural networks structure, iterate over it to find a better network that suits our needs. But what are other ways to improve your accuracy ? One way is to use ensemble methods i.e. train may networks independently, predict output on each of them, and then return the average of those outputs. What other way ? We can use different parts of images as input and use the average of those outputs - one of the ways is 10-crop i.e. run 10 different sub-images on the network and average the output. However, these methods are not practically useful in real-time applications or let's say the tradeoff between computational/time resources isn't worth the accuracy improvement these provide in such cases.
