import torch
import numpy as np

from wsl.saliency.misc_functions import get_example_params, convert_to_grayscale, save_gradient_images


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


class VanillaBackprop():
	"""
		Produces gradients generated with vanilla back propagation from the image
	"""
	def __init__(self, model):
		self.model = model.to(device)
		self.gradients = None
		# Put model in evaluation mode
		self.model.eval()
		# self.model.inputs.requires_grad = True
		# Hook the first layer to get the gradient
		# self.hook_layers()


	def hook_input(self, input_tensor):
		def hook_function(grad_in):
			self.gradients = grad_in
		input_tensor.register_hook(hook_function)


	# def hook_layers(self):
	# 	def hook_function(module, grad_in, grad_out):
	# 		self.gradients = grad_in[0]

	# 	def printnorm(self, input, output):
	# 		# input is a tuple of packed inputs
	# 		# output is a Tensor. output.data is the Tensor we are interested
	# 		print('Inside ' + self.__class__.__name__ + ' forward')
	# 		print('')
	# 		print('input: ', type(input))
	# 		print('input[0]: ', type(input[0]))
	# 		print('output: ', type(output))
	# 		print('')
	# 		print('input size:', input[0].size())
	# 		print('output size:', output.data.size())
	# 		print('output norm:', output.data.norm())

	# 	def printgradnorm(module, grad_input, grad_output):
	# 		print('Inside ' + self.__class__.__name__ + ' backward')
	# 		print('Inside class:' + self.__class__.__name__)
	# 		print('')
	# 		print('grad_input: ', type(grad_input))
	# 		print('grad_input[0]: ', type(grad_input[0]))
	# 		print('grad_output: ', type(grad_output))
	# 		print('grad_output[0]: ', type(grad_output[0]))
	# 		print('')
	# 		print('grad_input size:', grad_input[0].size())
	# 		print('grad_output size:', grad_output[0].size())
	# 		print('grad_input norm:', grad_input[0].norm())

	# 	# Register hook to the first layer
	# 	print(list(self.model.features._modules.items()))
	# 	first_layer = list(self.model.features._modules.items())[0][1]
	# 	first_layer.register_hook(hook_function)
	# 	first_layer.register_hook(printgradnorm)
	# 	first_layer.register_forward_hook(printnorm)


	def generate_gradients(self, input_image, target_class):
		# Forward
		input_image.requires_grad = True
		self.hook_input(input_image)
		print(input_image.size())
		# self.hook_input(input_image)
		model_output = self.model(input_image.to(device))
		print(model_output.size())
		# Zero grads
		self.model.zero_grad()
		# Target for backprop
		one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().to(device)
		one_hot_output[0] = 1
		# Backward pass
		model_output.backward(gradient=one_hot_output)
		# Convert Pytorch variable to numpy array
		# [0] to get rid of the first channel (1,3,224,224)
		gradients_as_arr = self.gradients.data.cpu().numpy()[0]
		print('shape ', np.shape(gradients_as_arr))
		
		return gradients_as_arr


if __name__ == '__main__':
	# Get params
	target_example = 1  # Snake
	(original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
		get_example_params(target_example)
	# Vanilla backprop
	VBP = VanillaBackprop(pretrained_model)
	# Generate gradients
	vanilla_grads = VBP.generate_gradients(prep_img, target_class)
	# Save colored gradients
	save_gradient_images(vanilla_grads, file_name_to_export + '_Vanilla_BP_color')
	# Convert to grayscale
	grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
	# Save grayscale gradients
	save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_Vanilla_BP_gray')
	print('Vanilla backprop completed')
