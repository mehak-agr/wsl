#!/usr/bin/python3
# +
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ast import literal_eval
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from torchsummary import summary

import json
import datetime
import cv2
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from wsl.saliency.misc_functions import get_example_params, save_class_activation_images
from wsl.locations import wsl_model_dir, wsl_summary_dir
from wsl.loaders.class_loaders import Loader
from wsl.networks.medinet.utils import box_to_map, rle2mask

from wsl.saliency import gradcam, guided_backprop, guided_gradcam, integrated_gradients, smooth_grad, vanilla_backprop

import matplotlib.pyplot as plt

def aupr(smap,bb_mask):
	def rint(mask):
		mask[mask > 0.5] = 1
		return mask

	fpr,tpr,thresholds = precision_recall_curve(rint(bb_mask.flatten()),smap.flatten())
	return auc(tpr,fpr)
	


# -

def main(name: str,
		 task: str,
		 store: bool):

	if name == 'all':
		models = wsl_model_dir.glob('*')
	else:
		models = wsl_model_dir.glob(f'*{name}*')
	models = list(models)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

	print('Number of potential model matches =', len(models))
	all_configs = []
	for m, path in enumerate(models):

		if 'debug' in str(path):  # Debugging models
			print('Debugging model')
			continue
		elif 'vgg19' not in str(path):  # Model is not wildcat
			print('Model is not densenet')
			continue
		elif 'wildcat' in str(path):  # Model is not wildcat
			print('Model is wildcat')
			continue
		elif not (path / 'configs.json').exists():  # Model not completed
			print('Model not completed')
			continue
		else:
			with open(path / 'configs.json') as f:
				configs = json.load(f)
				print(configs)
				# if configs['pretrained'] == False:
				# 	continue
			dataset = Loader(data=configs['data'],
							 split='test',
							 extension=configs['extension'],
							 classes=configs['classes'],
							 column=configs['column'],
							 regression=configs['regression'])

		# print(f'Model {m} : {path}')
		try:
			checkpoint = torch.load(path / 'best.pt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
		except:
			continue
		# checkpoint = torch.load(path / 'best.pt', map_location='cpu')
		checkpoint['model'] = checkpoint['model'].module
		checkpoint['model'].get_map = True
		checkpoint['model'].eval()
		# summary(checkpoint['model'],(3,224,224))
		org_size = (1024, 1024)
		new_size = (224, 224)
		sigmoid = torch.nn.Sigmoid().cuda()
		all_scores = defaultdict(list)

		VBP = vanilla_backprop.VanillaBackprop(checkpoint['model'])
		IG = integrated_gradients.IntegratedGradients(checkpoint['model'])
		GBP = guided_backprop.GuidedBackprop(checkpoint['model'])
		GCAM = gradcam.GradCam(checkpoint['model'],target_layer=34)
		# with torch.set_grad_enabled(False):
		print(len(dataset))
		for idx, data in tqdm(enumerate(dataset)):
			img, label = data
			name = dataset.names[idx]
			labels = dataset.labels[idx]

			saliency_label = 1

			for i, label in enumerate(labels):
				if label == 0:
					saliency_label = 0
					break

			if saliency_label == 0:
				continue
			# saliency_label = torch.tensor(saliency_label)
			saliency_label = torch.tensor(saliency_label).to(device)

			vanilla_grads = VBP.generate_gradients(img.unsqueeze(dim=0).cuda().float(), saliency_label)
			grayscale_vanilla_grads = vanilla_backprop.convert_to_grayscale(vanilla_grads)
			# print(np.shape(grayscale_vanilla_grads))
			# vanilla_backprop.save_gradient_images(grayscale_vanilla_grads, '/data/2015P002510/nishanth/WSL/wsl/wsl/Example_maps/GRAD')
			integrated_grads = IG.generate_integrated_gradients(img.unsqueeze(dim=0).cuda().float(), saliency_label, 100)
			grayscale_integrated_grads = integrated_gradients.convert_to_grayscale(integrated_grads)
			# vanilla_backprop.save_gradient_images(grayscale_integrated_grads, '/data/2015P002510/nishanth/WSL/wsl/wsl/Example_maps/IG')
			guided_grads = GBP.generate_gradients(img.unsqueeze(dim=0).cuda().float(), saliency_label)
			grayscale_guided_grads = guided_backprop.convert_to_grayscale(guided_grads)
			# vanilla_backprop.save_gradient_images(grayscale_guided_grads, '/data/2015P002510/nishanth/WSL/wsl/wsl/Example_maps/GBP')
			smooth_grad_mask = smooth_grad.generate_smooth_grad(VBP, img.unsqueeze(dim=0).cuda().float(), saliency_label, 5, 0.3)
			grayscale_smooth_grad = smooth_grad.convert_to_grayscale(smooth_grad_mask)

			smooth_grad_mask = smooth_grad.generate_smooth_grad(IG, img.unsqueeze(dim=0).cuda().float(), saliency_label, 5, 0.3)
			grayscale_smooth_ig = smooth_grad.convert_to_grayscale(smooth_grad_mask)
			cam = GCAM.generate_cam(img.unsqueeze(dim=0).cuda().float(), saliency_label)
			# grayscale_cam = guided_backprop.convert_to_grayscale(cam)

			cam_gb = guided_gradcam.guided_grad_cam(cam, guided_grads)
			grayscale_cam_gb = guided_gradcam.convert_to_grayscale(cam_gb)
			# vanilla_backprop.save_gradient_images(cam, '/data/2015P002510/nishanth/WSL/wsl/wsl/Example_maps/GCAM')

			# # Save mask2
			# save_class_activation_images(img, cam, '/data/2015P002510/nishanth/WSL/wsl/wsl/Example_maps/GCAM_color')
			# score = []
			# np.save('/data/2015P002510/nishanth/WSL/wsl/wsl/AUPRC_scores/{}_{}.npy'.format('GRAD','resnet18'),np.zeros((2,2))) #test

			for i, label in enumerate(labels):
				if label == 0:
					continue

				if task == 'detect':
					ground_map = box_to_map(dataset.df[dataset.df.Id == name].box.to_list(),
											np.zeros(org_size))
					ground_map = cv2.resize(ground_map, new_size, interpolation=cv2.INTER_NEAREST).clip(0, 1)

					all_scores['GRAD'].append(aupr(grayscale_vanilla_grads,ground_map))
					all_scores['SG'].append(aupr(grayscale_smooth_grad,ground_map))
					all_scores['IG'].append(aupr(grayscale_integrated_grads,ground_map))
					all_scores['SIG'].append(aupr(grayscale_smooth_ig,ground_map))
					all_scores['GBP'].append(aupr(grayscale_guided_grads,ground_map))
					all_scores['GCAM'].append(aupr(cam,ground_map))
					all_scores['GGCAM'].append(aupr(grayscale_cam_gb,ground_map))
					# all_scores['GRAD'].append(aupr(cv2.resize(grayscale_vanilla_grads, new_size, interpolation=cv2.INTER_AREA),ground_map))
				
				elif task == 'segment':
					ground_map = np.zeros(org_size)
					eps = dataset.df[dataset.df.Id == name].EncodedPixels.to_list()
					for ep in eps:
						ground_map += rle2mask(ep, np.zeros(org_size)).T
				else:
					print('Ground truth not available.')

		for key in all_scores.keys():
			print(key, ' ', np.mean(all_scores[key]))
			np.save('/data/2015P002510/nishanth/WSL/wsl/wsl/AUPRC_scores/{}_{}.npy'.format(key,'vgg_test'),all_scores[key])


		configs['wild'] = np.mean(all_scores)
		
		all_configs.append(configs)

	df = pd.DataFrame.from_dict(all_configs)
	print(df)
	time = datetime.datetime.now().strftime('%H_%d_%m')
	if store:
		df.to_csv(wsl_summary_dir / f'wild_{time}')


