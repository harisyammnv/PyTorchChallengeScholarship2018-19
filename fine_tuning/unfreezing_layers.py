'''Code snippets on how to unfreeze and train specific layers'''

### How to import a pretrained model from torchvision
import torchvision.models as models
model = models.resnet152()

### How to train the last layer only (classifier)
for param in model.parameters():
    param.requires_grad = False

### Training all the layers, basically retraining the whole network
### For that you need to unfreeze all the layers, and for all parameters set to True in order to allow backporpagation
for param in model.parameters():
    param.requires_grad = True

### Example: How to freeze all layers except for the last 3
### This is an example used on ResNet152 pre-trained model
for name, child in model.named_children():
    if name in ['conv1', 'bn1', 'relu', 'maxpool','layer1', 'layer2']:
      print(name + ' is frozen')
      for param in child.parameters():
          param.requires_grad = False
    else:
      print(name + ' is unfrozen')
      for param in child.parameters():
          param.requires_grad = True

#In order to make it work during training you'll also have to adjust your optimizer such as below
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0006, momentum=0.9, nesterov=True)
