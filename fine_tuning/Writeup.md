Resources/links


https://github.com/masterflorin/PyTorchChallengeScholarship2018-19/blob/master/fine_tuning/fine_tuning.md
https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
[1] https://arxiv.org/abs/1506.01186












Workflow
The first thing we do is State the problem
Using Transfer learning when the dataset size is small and the targets are not similar to the targets in the ImageNet Database
Then we mention the first step we state the problem of not reaching an accuracy higher than 89% or 91% on the validation set. Classic problem. We hit that wall too. 
Propose food for thought, and provide some examples from the repo and other resources
Data Augmentation
Learning rate
How should my classifier look like?
Why only Linear and Dropout layers in Classifiers?
We could add special Pooling layers and also Use BatchNorm1d
Optimizers
How to deal with the Learning Rate? 
How small should it be?
Use a scheduler to save yourself from doing everything manually
Unfreezing layers selectively and gradually (copied the explanation below should refactor it)
The reason why i mentioned not to unfreeze all the layers at once is the last conv layers are the layers which detect more richer representations in the image, since those richer repr are responsible for the classification you need to train them longer than the initial layers so this is like prioritizing ...therefore unfreeze the last conv layers/blocks train them with 10x-100x reduction of Learning rate then go to the next block reduce the LR by 10x-100x compared to the previous and then slowly move on till the starting layers

Regularization - Weight Decay
Final considerations

Draft

Intro/Overview
You trained your model on let’s say 20-30 epochs with some learning using Adam or SGD as an optimizer but your accuracy on the validation set stopped at 90% or below. What do you do? How do I get to that 98% or 99% accuracy? Is that even possible? Of course it is. Both Harry and I, hit the same wall as you did and fortunately we have managed to overcome the obstacle and got the best solution. In this article, we are going to teach you how to get to those magical results. 

ADD gif with MAGIC https://giphy.com/gifs/shia-labeouf-12NUbkX6p4xOO4
(Caption: It’s not magic though, you’ll get there through experimentation)

Food for thought
In the following paragraphs we’re going to present you some of the things that will definitely help you fine-tune your model


Data Augmentation
This is one of those parts where you really have to test and visualize how the image looks like.
It’s obviously a tricky task to get it right so let’s think about how we could go about it. 

Points to consider: 
Am I doing enough data augmentation? 
Am I doing too much? 

One of the easiest ways to go about it is to work the simple transforms from PyTorch like RandomRotation or ColorJitter. We should consider adding only 1-2 transform functions at a time, the reason for this is that the dataset we are dealing with is not very complex. Moreover, starting with fewer can help us identify which one worked best.

Code snippet
data_transform = transforms.Compose([
    	transforms.RandomRotation(25),
    	transforms.RandomResizedCrop(224),
    	transforms.ToTensor(),
    	transforms.Normalize([0.485, 0.456, 0.406],
                         	[0.229, 0.224, 0.225])])

If you are willing to spend some additional time on exploring different transformations consider the following Colab notebook as a source. This also covers more interesting data augmentations, not necessary for the project but worth exploring if you are up for the task.


Unfreezing layers selectively
Have you ever wondered when you were at lesson 5.41 and 5.42 that you could unfreeze different layers of the model and train them again? Not? Me neither. :)

Why am I bringing this up though? I’ve noticed that the ideta of unfreezing the whole model and retrain it again with a smaller learning rate to be quite popular but is it really that helpful? I experimented myself and got a very small boost if none at all. 

Let’s think about it why should we retrain the whole thing? Isn’t there a way to just retrain only the layers that bring the most value? That’s right. Remember from lesson 5 on CNNs that we learned how the layers stacks play a different role in how the features are captured by the model. In 5.42, we learn that the last layers see the more complex patterns of the images therefore that’s most likely where we our model is not doing well enough.This makes more sense, doesn’t it? Focusing our efforts in the right places in order to the get the right results.

Okay, but how do we actually do that?

From the repository on PyTorch Challenge Scholarship that I’m building I’m going to provide you some help on how to unfreeze only the last 2 layer stacks and retrain the model based on that.

Now, I’m going to take a ResNet architecture, specifically ResNet152 to check what are the names of the layer stacks in our model.

for name, child in model.named_children():
print(name)

It will print out the following:
# conv1
# bn1
# relu
# maxpool
# layer1
# layer2
# layer3
# layer4
# avgpool
# fc

We already know that in order to backpropagate through layers we have to set it to True
for param in model.parameters():
	param.requires_grad = True



What we can do is combine those two pieces of information in two and come up with the following piece of code.

for name, child in model.named_children():
	if name in ['layer3', 'layer4']:
  		print(name + ' is unfrozen')
  		for param in child.parameters():
      			param.requires_grad = True
	else:
  		print(name + ' is frozen')
  		for param in child.parameters():
      			param.requires_grad = False


What we’re doing above is unfreezing layer stacks 3 and 4, and leaving the rest frozen. Are we done? Not quite. In order to make this work, we have to adjust our optimizer to take which layers to work with.

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0006, momentum=0.9, nesterov=True)





Learning Rate
The learnimg rate is perhaps one of the most import hyperparameters which has to be set for enabling your deep neural network to perform better. 
Generally the Deep Neural networks are trained through backprop using optimizers like Adam, Stochastic Gradient Descent, Adadelta etc.. In all these optimizers the learning rate is the input parameter and it guides the optimizer through rough terrain of the Loss function.

The problems which the Optimizer could encounter:

If the learning rate is too small - training is more reliable, but optimization will take a lot of time because steps towards the minimum of the loss function are tiny.
If the learning rate is too big - then training may not converge or even diverge. Weight changes can be so big that the optimizer overshoots the minimum and makes the loss worse. 
Picture to show the overshoot

Best Approach to resolve this issue: Start from a Larger learning rate and gradually reduce them to smaller values or start from smaller and increase gradually...therefore making the optimizer reach its global minima. This approach is outlined in [1] and is implemented in fast.ai library (higher level API on PyTorch). We are just showing the usage of implemented function here:

learn.lr_find()
This function would train a network starting from a low learning rate and increase the learning rate exponentially for every batch.
2 Pictures to explain this and the selection of the learning rate….
This process has to be repeated everytime when we unfreeze some layers of the network


How should my classifier look like?
Generally in the transfer learning tasks the Fully Connected classifier layers are ripped off and new FC layers are added to train on the new data and perform the new task. But many would generall stick with the conventional Linear and Dropout layers in the FC classifiers… Could we add some different layers? Yes we could, consider the following example where we added AdaptivePooling Layers in the new classifier

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
In the above example we have added AdaptiveMaxPool2d and AdaptiveAveragePool2d and flattened them out and concatenated them to form a linear layer of size 2208*2. Why we did this is because, The Pooling layers capture richer features in the conv layers and we need to provide them as best as possible to the Classifier so they could classifiy easily and this would also effectively reduce the number of linear layers we need. This implementation is outlined is fast.ai library, we just re-implemented it here.
