## Fine-tuning torchvision models

Documentation with links towards specific topics regarding fine-tuning.

- [How to fine-tune a pretrained model - code snippets](unfreezing_layers.py)

### Food for thought on how to fine tune your pretrained model

From **@Harry007** on slack channel with additional edits.

Here are some questions:

1. Am I doing enough Data Augmentation? Am I doing too much? Is this is an imbalanced Dataset, how do we adjust it?

2. Why am I sticking with only Adam or SGD optimizers? Are there any more Optimizers to try out? (Yes there are check them out for e.g: Stochastic Gradient Descent with Restarts)

3. Should the classifier layers consists of Linear and Dropouts? Can we add BatchNorm1d? Can we add Special Pooling layers?

4. What about the Learning Rate schedulers? Are there any new LR schedulers? e.g. Cyclic LR scheduler

5. Try to unfreeze the layers gradually and everytime experiment with 2 to 4 different learning rates everytime you unfreeze a block or a layer ([Differential Learning rates](https://discuss.pytorch.org/t/different-learning-rates-within-a-model/1307) )

6. Use advanced regularization techniques like weight decay and if you are using SGD try to change the momentum values (Momentum Correction)

7. Which are the convolutional layers that bring the big $$$? Consider unfreezing only the last 2-3 layers, think about which layers are designed to retrieve the more complex patterns in the images.

8. Whenever I unfreeze a layer how small should I reduce the learning rate? How about experimenting with reducing by a factor 10x-100x?

Lastly, after trying out solutions for these questions you will understand that you do not need to unfreeze all the layers to reach 99% Val accuracy, so guys try these ideas out!!!! You can definitively achieve an accuracy on validation >99% in just 1.5 hours.


#### TODO:
- [ ] Create jupyter notebooks with specific examples and explanations on fine-tuning
- [ ] Update file to suggest resources for fine-tuning in PyTorch