'''Code snippets on how to unfreeze and train specific layers'''

### Check PyTorch version
import torch
print(torch.__version__)

### How to set the desired seed for random sampling
torch.manual_seed(seed)

### How to import a pretrained model from torchvision
import torchvision.models as models
model = models.resnet152(pretrained=True)
model = models.densenet161(pretrained=True)

### How to train the last layer only (classifier)
for param in model.parameters():
    param.requires_grad = False

### Training all the layers, basically retraining the whole network
### For that you need to unfreeze all the layers, and for all parameters set to True in order to allow backpropagation
for param in model.parameters():
    param.requires_grad = True

### Let's print the names of the layer stacks for our model
for name, child in model.named_children():
    print(name)

# for ResNet152 it should print
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

### Example: How to freeze all layers except for the last 3
### This is an example used on ResNet152 pre-trained model
### We can add which layers we want or not to a list for example
for name, child in model.named_children():
    if name in ['conv1', 'bn1', 'relu', 'maxpool','layer1', 'layer2']:
      print(name + ' is frozen')
      for param in child.parameters():
          param.requires_grad = False
    else:
      print(name + ' is unfrozen')
      for param in child.parameters():
          param.requires_grad = True

### Check what it's printing
# conv1 is frozen
# bn1 is frozen
# relu is frozen
# maxpool is frozen
# layer1 is frozen
# layer2 is frozen
# layer3 is unfrozen
# layer4 is unfrozen
# avgpool is unfrozen
# fc is unfrozen

### In order to unfreeze only a few layers on a DenseNet model we have to adjust the way we traverse the dense blocks
for name in model.children():
  for child, config in name.named_children():
    print(child)

# conv0
# norm0
# relu0
# pool0
# denseblock1
# transition1
# denseblock2
# transition2
# denseblock3
# transition3
# denseblock4
# norm5

### Now, let's adjust our method of unfreezing layers and unfreeze the last two blocks (this might be a bit much)
for name in model.children():
  for child, config in name.named_children():
    if child in ['denseblock4', 'denseblock3']:
      print(str(child) + ' is unfrozen')
      for param in config.parameters():
          param.requires_grad = True
    else:
      print(str(child) + ' is frozen')
      for param in config.parameters():
          param.requires_grad = False

### Check what it's printing
# conv0 is frozen
# norm0 is frozen
# relu0 is frozen
# pool0 is frozen
# denseblock1 is frozen
# transition1 is frozen
# denseblock2 is frozen
# transition2 is frozen
# denseblock3 is unfrozen
# transition3 is frozen
# denseblock4 is unfrozen
# norm5 is frozen

### In order to make it work during training you'll also have to adjust your optimizer as such
### We're using a SGD with momentum, and nesterov activated, nothing fancy
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0006, momentum=0.9, nesterov=True)

### How do we adjust the learning rate automatically? 
### for more examples: https://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate
from torch.optim import scheduler

scheduler = scheduler.StepLR(optimizer, step_size=30, gamma=0.1) ### Experiment with different step size and gamma factor
for epoch in range(100):
    scheduler.step()
    train(...)
    validate(...)

### After training, you can find out how many total parameters and training parameters, 
### this can give you an indication of the size of our model's checkpoint
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')