## Draft for fine-tuning article

## Table of contents

- Overview
- Data Augmentation
- How should my classifier look like?
- Learning rate
- How to deal with the Learning Rate after unfreezing the convolutional layers? 
- About optimizers
- Unfreezing layers selectively and gradually
- Regularization - Weight Decay
- Final considerations
- Resources and where to go next


## Overview

**Notes & Prequisites**: Before you start reading this article, we are assuming that you have already trained a pre-trained model and that you are looking for solutions on how to improve your model's ability to generalize. [Harry](https://github.com/harisyammnv) and I focused our writing on what you could apply to the Flower Classification as the final lab challenge of the Udacity's [PyTorch Challenge Scholarship] but while this is true you could definitively learn something if you're at the beginning of your Deep Learning journey. Let's begin!

Once upon a time, you trained your model on let’s say 20-30 epochs with some learning rate and using Adam or SGD as an optimizer but your accuracy on the validation set stopped at 90% or below. What do you do from here on? How do I get to that 98% or 99% accuracy? Is that even possible? Of course it is. Both Harry and I, hit the same wall as you did and fortunately we have managed to overcome the obstacle and 

With that said, in the following paragraphs we’re going to present you some of the things that will definitely help you fine-tune your model.

ADD gif with MAGIC https://giphy.com/gifs/shia-labeouf-12NUbkX6p4xOO4
(Caption: It’s not magic though, you’ll get there through experimentation)

## 1) Data Augmentation
This is one of those parts where you really have to test and visualize how the image looks like.
It’s obviously a tricky task to get it right so let’s think about how we could go about it. 

- Points to consider: 
- Am I doing enough data augmentation? 
- Am I doing too much? 

One of the easiest ways to go about it is to work the simple transforms from PyTorch like RandomRotation or ColorJitter. We should consider adding only 1-2 transform functions at a time, the reason for this is that the dataset we are dealing with is not very complex. Moreover, starting with fewer can help us identify which one worked best.

An Example: 
~~~
data_transform = transforms.Compose([
    	transforms.RandomRotation(25),
    	transforms.RandomResizedCrop(224),
    	transforms.ToTensor(),
    	transforms.Normalize([0.485, 0.456, 0.406],
                         	[0.229, 0.224, 0.225])])
~~~
If you are willing to spend some additional time on exploring different transformations consider the following [Colab notebook](https://colab.research.google.com/drive/109vu3F1LTzD1gdVV6cho9fKGx7lzbFll#scrollTo=rQ6DFPNvVD8s) as a source. This also covers more interesting data augmentations, not necessary for the project but worth exploring if you are up for the task.

## 2) Learning Rate

The learnimg rate is perhaps one of the most import hyperparameters which has to be set for enabling your deep neural network to perform better on train/val datasets. Generally the Deep Neural networks are trained through back-propagation using optimizers like Adam, Stochastic Gradient Descent, Adadelta etc. In all of these optimizers the learning rate is an input parameter and it guides the optimizer through rough terrain of the Loss function. 

The problems which the Optimizer could encounter are:

- If the learning rate is too small - training is more reliable, but optimization will take a lot of time because steps towards the minimum of the loss function are tiny.
- If the learning rate is too big - then training may not converge or even diverge. Weight changes can be so big that the optimizer overshoots the minimum and makes the loss worse. 

![Lr](https://github.com/harisyammnv/MicrosoftMalwareChallenge2018/blob/master/1.PNG)

[Image Credit](https://www.jeremyjordan.me/nn-learning-rate/)

**Best Approach to find optimal Initial Learning rate**: Start from a larger learning rate and gradually reduce them to smaller values or start from smaller and increase gradually after traversing through each mini-batch. This approach is outlined in [1](https://arxiv.org/abs/1506.01186) and is implemented in [fast.ai](https://www.fast.ai/) library (A higher level API for PyTorch). We are just showing the usage of implemented function [3](https://github.com/fastai/fastai/blob/master/old/fastai/learner.py) here:

~~~ 
learn.lr_find()
~~~

**More Info: This function would train a network starting from a low learning rate and increase the learning rate exponentially for every batch. This process has to be repeated everytime we unfreeze some layers of the network.**

In the Flower Classification Dataset if we use `lr_find()` for a `densenet161` with all the layers frozen except the classifier the results are as follows:

![](https://github.com/harisyammnv/MicrosoftMalwareChallenge2018/blob/master/4.PNG) ![](https://github.com/harisyammnv/MicrosoftMalwareChallenge2018/blob/master/3.PNG)

**Observation** The optimal initial learning rate for densenet could be in the range marked by red dotted lines, but we selected `2e-2`. Generally the Learning rate is selected where there is maximum decrement of  loss function is observed 

## 3) How should my classifier look like?
Generally in the transfer learning tasks the Fully Connected (FC) classifier layers are ripped off and new FC layers are added to train on the new data and perform the new task. But many would generally stick with the conventional Linear and Dropout layers in the FC layers. Could we add some different layers? Yes we could, consider the following example where we added AdaptivePooling Layers in the new classifier

~~~~
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
      
class ClassifierNew(nn.Module):
    def __init__(self, inp = 2208, h1=1024, out = 102, d=0.35):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d((1,1))
        self.mp = nn.AdaptiveMaxPool2d((1,1))
        self.fla = Flatten()
        self.bn0 = nn.BatchNorm1d(inp*2,eps=1e-05, momentum=0.1, affine=True)
        self.dropout0 = nn.Dropout(d)
        self.fc1 = nn.Linear(inp*2, h1)
        self.bn1 = nn.BatchNorm1d(h1,eps=1e-05, momentum=0.1, affine=True)
        self.dropout1 = nn.Dropout(d)
        self.fc2 = nn.Linear(h1, out)
        
    def forward(self, x):
        ap = self.ap(x)
        mp = self.mp(x)
        x = torch.cat((ap,mp),dim=1)
        x = self.fla(x)
        x = self.bn0(x)
        x = self.dropout0(x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)         
        x = self.fc2(x)
        
        return x
	
~~~~
![Lr12](https://github.com/harisyammnv/MicrosoftMalwareChallenge2018/blob/master/12.PNG)

In the above example we have added AdaptiveMaxPool2d and AdaptiveAveragePool2d and flattened them out and concatenated them to form a linear layer of size `-1 x 2* size(Last BatchNorm2d Layer)`. 

For example in `DenseNet161`:
- The last BacthNorm2d layer has an output dimension of `-1x2208x7x7`  
- After passing the mini-batch through the 2 Adaptive Pooling layers we obtain 2 output tensors of shape `-1x2208x1x1` 
- Concatenation of the above 2 tensors would result in a tensor of shape `-1x4416x1x1`  
- Finally flattening the tensor of shape `-1x4416x1x1` would result a Liner Layer of `-1x4416` i.e `(-1x2*(2208))`
- This layer is then connected to the Fully Connected part
- ***Note:*** -1 in the above tensor shapes should be replaced with the mini-batch size

**Reason:** Why we did this? It could be attributed to The Pooling layers because they capture richer features from the conv layers and we need to provide them as best as possible to the Classifier so they could classifiy easily and this would also effectively reduce the number of linear layers we need. This implementation is outlined is [fast.ai](https://www.fast.ai/) library, we just re-implemented it here.

## 4) Learning Rate Annealing / Scheduling
The key idea here is to iteratively reduce the learning rate after every few epochs so that the optimizer would reach global/best local optima without any huge oscillations. The most popular form of learning rate annealing is a step decay where the learning rate is reduced by some percentage after a set number of training epochs. The other common scheduler is ReduceLRonPlateau. But here we would like to highlight a new one which was highlighted in [1](https://arxiv.org/abs/1506.01186) and was termed as cyclic learning rates.

![Lr2](https://github.com/harisyammnv/MicrosoftMalwareChallenge2018/blob/master/2.PNG)

[Image Credit](https://www.jeremyjordan.me/nn-learning-rate/)

The intution behind why this learning rate annealing improves the val accuracy is outlined in [2](https://www.jeremyjordan.me/nn-learning-rate/). An exemplary cyclicLR class is taken from [4](https://github.com/pytorch/pytorch/pull/2016) & [5](https://github.com/thomasjpfan/pytorch/blob/401ec389db2c9d2978917a6e4d1101b20340d7e7/torch/optim/lr_scheduler.py) and is given below:

~~~
def cyclical_lr(step_sz, min_lr=0.001, max_lr=1, mode='triangular', scale_func=None, scale_md='cycles', gamma=1.):
    if scale_func == None:
        if mode == 'triangular':
            scale_fn = lambda x: 1.
            scale_mode = 'cycles'
        elif mode == 'triangular2':
            scale_fn = lambda x: 1 / (2.**(x - 1))
            scale_mode = 'cycles'
        elif mode == 'exp_range':
            scale_fn = lambda x: gamma**(x)
            scale_mode = 'iterations'
        else:
            raise ValueError(f'The {mode} is not valid value!')
    else:
        scale_fn = scale_func
        scale_mode = scale_md

    lr_lambda = lambda iters: min_lr + (max_lr - min_lr) * rel_val(iters, step_sz, scale_mode)

    def rel_val(iteration, stepsize, mode):
        cycle = math.floor(1 + iteration / (2 * stepsize))
        x = abs(iteration / stepsize - 2 * cycle + 1)
        if mode == 'cycles':
            return max(0, (1 - x)) * scale_fn(cycle)
        elif mode == 'iterations':
            return max(0, (1 - x)) * scale_fn(iteration)
        else:
            raise ValueError(f'The {scale_mode} is not valid value!')

    return lr_lambda

optimizer = optim.SGD(model.parameters(), lr=1.)
clr = cyclical_lr(step_size, min_lr=0.001, max_lr=1, mode='triangular2')
scheduler = lr_scheduler.LambdaLR(optimizer, [clr])
scheduler.step()
optimizer.step()
~~~

## 5) Optimizer
After using the conventional Adam/SGD we had tried a new Optimizer which was cosine Cyclic Learning Rate annealing in combination with SGD which is termed Stochastic Gradient Descent with Restarts (SGDR) which proved to be better. SGDR is an aggressive annealing schedule which is combined with periodic "restarts" to the original starting learning rate. The LRsheduler for the Flower classification dataset looks as the one below and we used it from the [fast.ai](https://www.fast.ai/) library because - [SGDR](https://github.com/fastai/fastai/blob/master/old/fastai/sgdr.py)  is still an open request [6](https://github.com/pytorch/pytorch/issues/3790), [7](https://github.com/allenai/allennlp/issues/1642), [8](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR) in PyTorch:

In Pytorch the Cosine Annealing Scheduler can be used as follows but it is without the restarts
~~~
## Only Cosine Annealing here
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
~~~

For the Flower dataset we used it from [fast.ai](https://www.fast.ai/) as following:
~~~
# SGDR is implemented as the optimizer if the cycle_len flag is provided to the fit method
learn.fit(lr, nr_of_epochs, cycle_len=1)
~~~

![Lr3](https://github.com/harisyammnv/MicrosoftMalwareChallenge2018/blob/master/6.PNG)

~~~
# SGDR + If Cycle Multipicity is needed
learn.fit(lr, nr_of_epochs, cycle_len=1, cycle_mult = 2)
~~~
The cycle multiplicity option in simpler words is basically increasing the LR decay over more number of epochs after every restart
![Lr4](https://github.com/harisyammnv/MicrosoftMalwareChallenge2018/blob/master/5.PNG)

**Observation:** By using these techniques the Validation loss has reduced significantly in the flower dataset and thereby increasing the val accuracy 

## 6) Unfreezing layers selectively
Have you ever wondered when you were at lesson 5.41 and 5.42 that you could unfreeze different layers of the model and train them again? Not? Me neither. :)

Why are we bringing this up though? We’ve noticed that the idea of unfreezing the whole model and retrain it again with a smaller learning rate to be quite popular but is it really that helpful? We've experimented ourselves and got a very small boost if none at all. Which doesn't mean that it's bad approach, what we suggest is another way, a more focused approach.

Let’s think about it why should we retrain the whole thing? Isn’t there a way to just retrain only the layers that bring the most value? That’s right. Remember from lesson 5 on CNNs that we learned how the layers stacks play a different role in how the features are captured by the model. In 5.42, we learn that the last layers see the more complex patterns of the images therefore that’s most likely where we our model is not doing well enough.This makes more sense, doesn’t it? Focusing our efforts in the right places in order to the get the right results.

Okay, but how do we actually do that?

From the repository on PyTorch Challenge Scholarship that I’m building I’m going to provide you some help on how to unfreeze only the last 2 layer stacks and retrain the model based on that.

Now, I’m going to take a ResNet architecture, specifically ResNet152 to check what are the names of the layer stacks in our model.

~~~
for name, child in model.named_children():
    print(name)

# It will print the names of the model components

conv1
bn1
relu
maxpool
layer1
layer2
layer3
layer4
avgpool
fc

~~~

We already know that in order to backpropagate through layers we have to set it to True

~~~
for param in model.parameters():
	param.requires_grad = True
~~~

What we can do is combine those two pieces of information to obtain the following:

~~~
for name, child in model.named_children():
	if name in ['layer3', 'layer4']:
  		print(name + ' is unfrozen')
  		for param in child.parameters():
      			param.requires_grad = True
	else:
  		print(name + ' is frozen')
  		for param in child.parameters():
      		    param.requires_grad = False
~~~

What we’re doing above is unfreezing layer stacks 3 and 4, and leaving the rest frozen. Are we done? Not quite. In order to make this work, we have to adjust our optimizer to work correctly on the selected layers.

~~~
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0006, momentum=0.9, nesterov=True)
~~~

**Explanation**: The reason why we mentioned not to unfreeze all the layers at once is the last convolutional layers are the layers which detect more richer representations in the image, since those richer representations are responsible for the classification you need to train them longer than the initial layers so this is like prioritizing. 

**Tip/Trick**: Unfreezeing the last conv layers/blocks and training them with 10x-100x reduction of learning rate then go to the next block reduce the LR by 10x-100x compared to the previous and then slowly move on till the starting layers.

**Observation:** For the Flower Classification Dataset when using `densenet161` and unfreezing the complete `densenetblock4` we have run the `lr_find()` from `Step-4` and observed that the learning rate has to be reduced by a factor of 50x for the `densenetblock4` and for the classifier we reduced by 2x

![Lr5](https://github.com/harisyammnv/MicrosoftMalwareChallenge2018/blob/master/7.PNG)


## 7) Weight Decay
Why do we only decay the learning rate? is it also possible to decay weights? Yes it is possible by employing L1/L2 regularization to the loss function. For ex if we have a cost function `E(w)` Gradient descent tells us to modify the weights `w` in the direction of steepest descent in `E` by the formula:

![Lr6](https://github.com/harisyammnv/MicrosoftMalwareChallenge2018/blob/master/8.PNG)

But if we add L2 regularization terms to the cost function, Gradient descent on the updated cost function will lead us to a weight decay term which will penalizes large weights and effectively limits the freedom in your model [9](https://stats.stackexchange.com/questions/29130/difference-between-neural-net-weight-decay-and-learning-rate).
The modified cost function and the modified weights update rule is given below

![Lr7](https://github.com/harisyammnv/MicrosoftMalwareChallenge2018/blob/master/10.PNG)

![Lr8](https://github.com/harisyammnv/MicrosoftMalwareChallenge2018/blob/master/9.PNG)

The term  

![Lr7](https://github.com/harisyammnv/MicrosoftMalwareChallenge2018/blob/master/11.PNG) 

causes the weights to decay in proportion to its size

In pytorch the weight decay could be implemented as follows
~~~
# similarly for SGD as well
torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
~~~

## Final considerations
All in all, for us, this was quite a difficult topic to tackle as fine-tuning a model is a very broad and challenging topic. Most of our efforts have been directed towards the Flower Classifier application as part of the PyTorch Challenge but some of the advice will certainly help you go further than that as they are pretty general. Through this, we hope that you found this article helpful.

## Resources/links

1. https://arxiv.org/abs/1506.01186
2. https://www.jeremyjordan.me/nn-learning-rate/
3. https://github.com/fastai/fastai/blob/master/old/fastai/learner.py
4. https://github.com/pytorch/pytorch/pull/2016
5. https://github.com/thomasjpfan/pytorch/blob/401ec389db2c9d2978917a6e4d1101b20340d7e7/torch/optim/lr_scheduler.py
6. https://github.com/pytorch/pytorch/issues/3790
7. https://github.com/allenai/allennlp/issues/1642
8. https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR
9. https://stats.stackexchange.com/questions/29130/difference-between-neural-net-weight-decay-and-learning-rate

## Where to go next:

- https://github.com/masterflorin/PyTorchChallengeScholarship2018-19/blob/master/fine_tuning/fine_tuning.md
- https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
- https://github.com/harisyammnv/PyTorchChallengeScholarship2018-19/blob/master/Pytorch_Challenge_Sample_Fastai.ipynb
