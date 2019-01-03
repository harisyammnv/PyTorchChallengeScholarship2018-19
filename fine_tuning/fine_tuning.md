## Fine-tuning torchvision models

### Food for thought on how to fine tune your pretrained model

From **[@Harry007](https://github.com/harisyammnv)** from slack channel with additional edits.

1. Why do I keep getting so many different results with the same configuration? Consider using `torch.manual_seed(value)` in order to help you reproduce your results. 
2. Am I doing enough Data Augmentation? Am I doing too much? Is this is an imbalanced Dataset, how do we adjust it?
	- Consider visualizing each data augmentation set. Have a look at this [Colab](https://colab.research.google.com/drive/109vu3F1LTzD1gdVV6cho9fKGx7lzbFll#scrollTo=rQ6DFPNvVD8s) 
	- Consider applying only 1-2 data augmentation functions at a time

3. Why am I sticking with only Adam or SGD optimizers? Are there any more Optimizers to try out? (Yes there are check them out for e.g: Stochastic Gradient Descent with Restarts)

2. Why am I sticking with only Adam or SGD optimizers? Are there any more Optimizers to try out? (Yes there are check them out for e.g: [Stochastic Gradient Descent with Restarts](https://medium.com/38th-street-studios/exploring-stochastic-gradient-descent-with-restarts-sgdr-fa206c38a74e), [More on SGDR](https://towardsdatascience.com/https-medium-com-reina-wang-tw-stochastic-gradient-descent-with-restarts-5f511975163))

3. Should the classifier layers consists of Linear and Dropouts? Can we add BatchNorm1d? Can we add Special Pooling layers? [Adaptive Avg Pool](https://forums.fast.ai/t/ideas-behind-adaptive-max-pooling/12634)

4. What about the Learning Rate schedulers? Are there any new LR schedulers? e.g. [CyclicLR scheduler](https://github.com/thomasjpfan/pytorch/blob/401ec389db2c9d2978917a6e4d1101b20340d7e7/torch/optim/lr_scheduler.py) 

7. Use advanced regularization techniques like weight decay and if you are using SGD try to change the momentum values (Momentum Correction)
	- Check the [documentation](https://pytorch.org/docs/master/optim.html#torch.optim.SGD) on how to apply weight decay and momentum to SGD
	- There is also an example in the [code snippets](unfreezing_layers.py)

8. Which are the convolutional layers that bring the most value? Consider unfreezing only the last 2-3 layers, think about which layers are designed to retrieve the more complex patterns in the images.
	- Learn how to do this by checking the [code snippets](unfreezing_layers.py).

9. Whenever I unfreeze a layer how small should I reduce the learning rate? How about experimenting with reducing by a factor 10x-100x?

10. Does using optim.SGD mean that I will get better accuracy? Experiment with both and see which one gives you the best results. Keep in mind though that it might not matter that much for this project, they are both very, very good. 
	- If you're really into this here are a few recommendations: [an overview](http://ruder.io/optimizing-gradient-descent/), [blog post](https://shaoanlu.wordpress.com/2017/05/29/sgd-all-which-one-is-the-best-optimizer-dogs-vs-cats-toy-experiment/).

Lastly, after trying out solutions for these questions you will understand that you do not need to unfreeze all the layers to reach 99% Val accuracy, so guys try these ideas out!!!! You can definitively achieve an accuracy on validation >99% in just 1.5 hours.


### Examples of fine-tuning with code:

- [How to fine-tune a pretrained model in code snippets](unfreezing_layers.py)


#### TODO:
- [ ] Organize the repo into basic and more advanced tips/techniques
- [ ] Create jupyter notebooks with specific examples and explanations on fine-tuning
- [ ] Update file to suggest resources for fine-tuning in PyTorch
- [ ] To add a table of contents
