# -*- coding:utf-8 -*-

import os
import torch
from torch.autograd import Variable
from net import NestFuse_autoencoder
from scipy.misc import imread, imsave, imresize
import utils
from args_fusion import args
import numpy as np
import copy


def load_model(path, deepsupervision):        #加载模型
	input_nc = 1
	output_nc = 1
	nb_filter = [64, 64, 64, 64, 256]

	nest_model = NestFuse_autoencoder(nb_filter, input_nc, output_nc, deepsupervision)
	nest_model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

	nest_model.eval()
	nest_model.cuda()

	return nest_model


def run_demo(nest_model, infrared_path, visible_path, output_path_root, index, f_type):
	img_ir, h, w, c = utils.get_test_image(infrared_path)
	img_vi, h, w, c = utils.get_test_image(visible_path)

	# prob_ir = pixel_probability(img_ir)
	# img_ir_add = probability_to_histogram(img_ir, prob_ir)
	# img_ir_add = torch.from_numpy(img_ir_add).float()
	# img_ir_add = img_ir_add.cuda()
	# img_ir_add = Variable(img_ir_add, requires_grad=False)
	#
	# prob_vi = pixel_probability(img_vi)
	# img_vi_add = probability_to_histogram(img_vi, prob_vi)
	# img_vi_add = torch.from_numpy(img_vi_add).float()
	# img_vi_add = img_vi_add.cuda()
	# img_vi_add = Variable(img_vi_add, requires_grad=False)

	if args.cuda:
		img_ir = torch.from_numpy(img_ir).float()
		img_ir = img_ir.cuda()
		img_vi = torch.from_numpy(img_vi).float()
		img_vi = img_vi.cuda()
	img_ir = Variable(img_ir, requires_grad=False)
	img_vi = Variable(img_vi, requires_grad=False)

	# utils.save_image_test_temp(img_ir-img_ir_add, './outputs/1.png')
	# utils.save_image_test_temp(img_ir_add, './outputs/2.png')
	# utils.save_image_test_temp(img_vi-img_vi_add, './outputs/3.png')
	# utils.save_image_test_temp(img_vi_add, './outputs/4.png')
	# print(c)

	if args.cuda:
		img_ir = img_ir.cuda()
		img_vi = img_vi.cuda()
	img_ir = Variable(img_ir, requires_grad=False)
	img_vi = Variable(img_vi, requires_grad=False)

	# dim = img_ir.shape
	if c is 1:           #图像不够大，没有分块
		if args.cuda:
			img_ir = img_ir.cuda()
			img_vi = img_vi.cuda()
		img_ir = Variable(img_ir, requires_grad=False)
		img_vi = Variable(img_vi, requires_grad=False)
		# encoder
		en_r = nest_model.encoder(img_ir)
		en_v = nest_model.encoder(img_vi)
		# fusion
		f = nest_model.fusion(en_r, en_v, f_type)

		# 保存中间层的特征图
		# decoder
		img_fusion_list = nest_model.decoder_eval(f)

	else:         #图像分块了
		# fusion each block
		img_fusion_blocks = []
		for i in range(c):
			# encoder
			img_vi_temp = img_vi[i]
			img_ir_temp = img_ir[i]
			if args.cuda:
				img_vi_temp = img_vi_temp.cuda()
				img_ir_temp = img_ir_temp.cuda()
			img_vi_temp = Variable(img_vi_temp, requires_grad=False)
			img_ir_temp = Variable(img_ir_temp, requires_grad=False)

			en_r = nest_model.encoder(img_ir_temp)
			en_v = nest_model.encoder(img_vi_temp)
			# fusion
			f = nest_model.fusion(en_r, en_v, f_type)
			# decoder
			img_fusion_temp = nest_model.decoder_eval(f)
			img_fusion_blocks.append(img_fusion_temp)
		img_fusion_list = utils.recons_fusion_images(img_fusion_blocks, h, w)

	############################ multi outputs ##############################################
	output_count = 0
	for img_fusion in img_fusion_list:
		file_name = 'fusion_nestfuse' + '_' + str(index) + '_subnet_' + str(output_count) + '_' + f_type + '.png'
		output_path = output_path_root + file_name
		output_count += 1
		# save images
		utils.save_image_test(img_fusion, output_path)
		print(output_path)


def main():
	# run demo
	test_path = "images/IV_images/"
	deepsupervision = False  # true for deeply supervision
	fusion_type = ['attention_avg', 'attention_max', 'attention_nuclear']

	with torch.no_grad():
		if deepsupervision:
			model_path = args.model_deepsuper
		else:
			model_path = args.model_default
		model = load_model(model_path, deepsupervision)
		# for j in range(3):
		# for j in range(3):
		for j in range(1):
			output_path = './outputs/' + fusion_type[j]

			if os.path.exists(output_path) is False:
				os.mkdir(output_path)
			output_path = output_path + '/'

			f_type = fusion_type[j]
			print('Processing......  ' + f_type)

			for i in range(50):
				index = i + 1
				infrared_path = test_path + 'IR' + str(index) + '.png'
				visible_path = test_path + 'VIS' + str(index) + '.png'
				run_demo(model, infrared_path, visible_path, output_path, index, f_type)
			# index = 2
			# infrared_path = test_path + 'IR' + str(index) + '.png'
			# visible_path = test_path + 'VIS' + str(index) + '.png'
			# run_demo(model, infrared_path, visible_path, output_path, index, f_type)

	print('Done......')

def pixel_probability(img):
	"""
	计算像素值出现概率
	:param img:
	:return:
	"""
	assert isinstance(img, np.ndarray)
	prob = np.zeros(shape=(256))
	for ri in img[0, 0]:
		for ci in ri:
			prob[ci] += 1
	r = img.shape
	prob = prob / (r[2] * r[3])
	return prob

def probability_to_histogram(img, prob):
	"""
	根据像素概率将原始图像直方图均衡化
	:param img:
	:param prob:
	:return: 直方图均衡化后的图像
	"""
	prob = np.cumsum(prob)  # 累计概率
	img_map = [int(255 * prob[i]) for i in range(256)]  # 像素值映射
	img_add = copy.deepcopy(img)
   # 像素值替换
	assert isinstance(img, np.ndarray)
	r = img.shape
	for ri in range(r[2]):
		for ci in range(r[3]):
			img_add[0, 0, ri, ci] = img_map[img[0,0, ri, ci]]
	return img_add


if __name__ == '__main__':
	main()
