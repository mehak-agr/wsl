import torch
from torch.nn import ReLU
from torch.autograd import Variable
import numpy as np
import cv2
from wsl.networks.medinet.utils import convert_to_grayscale


class BackProp():
    """
        Produces gradients generated with integrated gradients from the image
    """
    def __init__(self, model, GBP = False):
        self.model = model
        self.gradients = None
        self.model.eval()
        self.GBP = GBP
        
        if self.GBP:
            self.forward_relu_outputs = []
            self.update_relus()
            self.hook_first()
        
    def hook_first(self):
        def hook_function(module, grad_in, grad_out):
            self.gradient = grad_in[0]
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

    def generate_gradients(self, img):
        
        # Forward
        img.requires_grad = True
        model_output = self.model(img)
        
        # Zero grads
        self.model.zero_grad()

        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().cuda()
        one_hot_output[0] = 1
        
        # Backward pass
        model_output.backward(gradient=one_hot_output)

        if self.GBP:
            return convert_to_grayscale(self.gradient.data.squeeze().cpu().numpy())
        else:
            return convert_to_grayscale(img.grad.data.squeeze().cpu().numpy())

    def generate_integrated_gradients(self, img, steps):
        xbar_list = self.generate_images_on_linear_path(img, steps)
        integrated_grads = np.zeros((img.shape[-2], img.shape[-1]))
        
        for xbar_image in xbar_list:
            single_integrated_grad = self.generate_gradients(xbar_image)
            # Add rescaled grads from xbar images
            integrated_grads += single_integrated_grad / steps
            
        return integrated_grads
    
    def generate_smooth_grad(self, img, param_n, param_sigma_multiplier):
        # Generate an empty image/matrix
        smooth_grad = np.zeros((img.shape[-2], img.shape[-1]))

        mean = 0
        sigma = param_sigma_multiplier / (torch.max(img) - torch.min(img)).item()
        for x in range(param_n):
            noise = Variable(img.data.new(img.size()).normal_(mean, sigma**2))
            noisy_img = img + noise
            grads = self.generate_gradients(noisy_img)
            smooth_grad += grads

        smooth_grad = smooth_grad / param_n
        return smooth_grad
    
    def generate_cam(self, img):
        img.requires_grad = True
        output, feat_map, _, handle = self.model(img)
        
        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_().cuda()
        one_hot_output[0] = 1
        self.model.zero_grad()
        output.backward(gradient=one_hot_output, retain_graph=True)
        
        feat_map = feat_map.squeeze(dim=0).data.cpu().numpy()
        guided_gradients = self.model.gradient.squeeze(dim=0).data.cpu().numpy()
        weights = np.mean(guided_gradients, axis=(1, 2))
        
        handle.remove()
        
        cam = np.ones(feat_map.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * feat_map[i, :, :]
        cam = cv2.resize(cam, (img.shape[-1], img.shape[-2]), interpolation=cv2.INTER_AREA)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        return cam
