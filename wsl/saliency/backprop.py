import torch
from torch.nn import ReLU
from torch.autograd import Variable
import numpy as np

from wsl.saliency.misc_functions import convert_to_grayscale


class Gradient():
    """
        Produces gradients generated with integrated gradients from the image
    """
    def __init__(self, model, GBP):
        self.model = model
        self.gradients = None
        self.model.eval()
        self.GBP = GBP
        
        if self.GBP:
            self.forward_relu_outputs = []
            self.update_relus()
            self.hook_first()

    def hook_input(self, input_tensor):
        def hook_function(grad_in):
            self.gradients = grad_in
        input_tensor.register_hook(hook_function)
        
    def hook_first(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)


    def generate_images_on_linear_path(self, input_image, steps):
        step_list = np.arange(steps + 1) / steps
        xbar_list = [input_image * step for step in step_list]
        return xbar_list

    def generate_gradients(self, img, target_class):
        
        # Forward
        img.requires_grad = True
        if not self.GBP:
            self.hook_input(img)
        model_output = self.model(img)
        
        # Zero grads
        self.model.zero_grad()

        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().cuda()
        one_hot_output[0] = 1
        
        # Backward pass
        model_output.backward(gradient=one_hot_output)

        return convert_to_grayscale(self.gradients.data.squeeze().cpu().numpy()).squeeze()

    def generate_integrated_gradients(self, img, target_class, steps):
        xbar_list = self.generate_images_on_linear_path(img, steps)
        integrated_grads = np.zeros((img.shape[-2], img.shape[-1]))
        
        for xbar_image in xbar_list:
            single_integrated_grad = self.generate_gradients(xbar_image, target_class)
            # Add rescaled grads from xbar images
            integrated_grads += single_integrated_grad / steps
            
        return integrated_grads
    
    def generate_smooth_grad(self, img, target_class, param_n, param_sigma_multiplier, steps):
        # Generate an empty image/matrix
        smooth_grad = np.zeros((img.shape[-2], img.shape[-1]))

        mean = 0
        sigma = param_sigma_multiplier / (torch.max(img) - torch.min(img)).item()
        for x in range(param_n):
            noise = Variable(img.data.new(img.size()).normal_(mean, sigma**2))
            noisy_img = img + noise
            if steps == 0:
                grads = self.generate_gradients(noisy_img, target_class)
            else:
                grads = self.generate_integrated_gradients(noisy_img, target_class, steps)
            smooth_grad += grads

        smooth_grad = smooth_grad / param_n
        return smooth_grad
