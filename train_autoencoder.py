# -*- coding:utf-8 -*-

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  #设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'

import sys
import time
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from net import NestFuse_autoencoder
from args_fusion import args
import pytorch_msssim
import copy


def main():
	original_imgs_path = utils.list_images(args.dataset)    #读取训练用图像，返回的是一个图像名的集合
	train_num = 4000        #训练用图像数量
	original_imgs_path = original_imgs_path[:train_num]     #
	# random.shuffle(original_imgs_path)            #图像随机排序
	for i in range(2,3):
		# i = 3
		train(i, original_imgs_path)


def train(i, original_imgs_path):

	batch_size = args.batch_size

	# load network model
	# nest_model = FusionNet_gra()
	input_nc = 1
	output_nc = 1
	deepsupervision = False  # true for deeply supervision
	# nb_filter = [64, 112, 160, 208, 256]            #每一层的输出特征图通道数
	nb_filter = [64, 64, 64, 64, 256]

	nest_model = NestFuse_autoencoder(nb_filter, input_nc, output_nc, deepsupervision)   #nest_model现在是一个model

	if args.resume is not None:           #
		print('Resuming, initializing using weight from {}.'.format(args.resume))     #format，用后面括号里的替换前面的
		nest_model.load_state_dict(torch.load(args.resume))       #加载新的model
	print(nest_model)        #输出新的model
	optimizer = Adam(nest_model.parameters(), args.lr)      #from torch.optim import Adam，优化器对象
	mse_loss = torch.nn.MSELoss()            #torch.nn.MSELoss()均方损失函数，loss(xi,yi)=(xi−yi)^2
	ssim_loss = pytorch_msssim.msssim         #结构相似性损失

	if args.cuda:          #初始值为1
		nest_model.cuda()

	tbar = trange(args.epochs)
	print('Start training.....')         #显示任务进度条，开始训练

	Loss_pixel = []      #像素损失
	Loss_ssim = []       #结构相似性损失
	Loss_all = []        #总损失
	count_loss = 0
	all_ssim_loss = 0.
	all_pixel_loss = 0.
	for e in tbar:
		print('Epoch %d.....' % e)       #显示任务进度条
		# load training database        #加载训练集
		image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)   #图像路径，batch数量
		nest_model.train()      #model训练
		count = 0
		for batch in range(batches):      #batches为batch数量，默认为1
			print(e,batch)
			image_paths = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
			img = utils.get_train_images_auto(image_paths, height=args.HEIGHT, width=args.WIDTH, flag=False)
			if e == 1:
				# 直方图均衡化
				prob = pixel_probability(img)
				img_add = probability_to_histogram(img, prob)

				img_add = torch.from_numpy(img_add).float()
				img_add = img_add.cuda()
				img_add = Variable(img_add, requires_grad=False)

			img = torch.from_numpy(img).float()
			count += 1
			optimizer.zero_grad()       #梯度置0
			img = Variable(img, requires_grad=False)

			if args.cuda:
				img = img.cuda()
			# get fusion image
			# encoder
			if e == 1:
				en = nest_model.encoder(img_add)
			else:
				en = nest_model.encoder(img)
			# decoder
			outputs = nest_model.decoder_train(en)
			# utils.save_image_test_temp(img, './outputs/1.png')
			# utils.save_image_test_temp(img_add, './outputs/2.png')
			# print(c)
			# resolution loss: between fusion image and visible image

			x = Variable(img.data.clone(), requires_grad=False)

			ssim_loss_value = 0.
			pixel_loss_value = 0.
			for output in outputs:  # 计算像素损失和结构相似性损失
				pixel_loss_temp = mse_loss(output, x)
				ssim_loss_temp = ssim_loss(output, x, normalize=True)
				ssim_loss_value += (1 - ssim_loss_temp)
				pixel_loss_value += pixel_loss_temp
			ssim_loss_value /= len(outputs)
			pixel_loss_value /= len(outputs)

			# total loss
			# total_loss = pixel_loss_value + args.ssim_weight[i] * ssim_loss_value
			total_loss = pixel_loss_value
			total_loss.backward()
			optimizer.step()      #更新模型

			all_ssim_loss += ssim_loss_value.item()        #item()，得到张量中的元素
			all_pixel_loss += pixel_loss_value.item()
			if (batch + 1) % args.log_interval == 0:            #batch是第几个batch，恰好是设置的args.log_interval的倍数
				mesg = "{}\t SSIM weight {}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t total: {:.6f}".format(  #\t,制表符，上下对齐。%.6f 输出小数,即保留小数点后6位
					time.ctime(), i, e + 1, count, batches,      #time.ctime()，当前时间
								  all_pixel_loss / args.log_interval,
								  (args.ssim_weight[i] * all_ssim_loss) / args.log_interval,
								  (args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval
				)#用像素损失，结构相似性损失，总损失分别除以args.log_interval
				# tbar.set_description(mesg)       #设置进度条名称
				Loss_pixel.append(all_pixel_loss / args.log_interval)
				Loss_ssim.append(all_ssim_loss / args.log_interval)
				Loss_all.append((args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval)
				count_loss = count_loss + 1
				all_ssim_loss = 0.
				all_pixel_loss = 0.

			if (batch + 1) % (200 * args.log_interval) == 0:
				# save model             训练一定数量batch后开始保存模型
				nest_model.eval()
				nest_model.cpu()
				save_model_filename = args.ssim_path[i] + '/' + "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
									  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[
										  i] + ".model"
				save_model_path = os.path.join(args.save_model_dir_autoencoder, save_model_filename)      #拼接路径
				torch.save(nest_model.state_dict(), save_model_path)       #保存模型参数字典
				# save loss data
				# pixel loss         i是结构相似性损失权重
				loss_data_pixel = Loss_pixel
				loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_pixel_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				scio.savemat(loss_filename_path, {'loss_pixel': loss_data_pixel})      #保存文件，loss_pixel是文件中的矩阵名，loss_data_pixel是数据
				# SSIM loss
				loss_data_ssim = Loss_ssim
				loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_ssim_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				scio.savemat(loss_filename_path, {'loss_ssim': loss_data_ssim})
				# all loss
				loss_data = Loss_all
				loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_all_epoch_" + str(e) + "_iters_" + \
									 str(count) + "-" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				scio.savemat(loss_filename_path, {'loss_all': loss_data})

				nest_model.train()
				nest_model.cuda()
				# tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

	#这是一次epoch完成之后
	# pixel loss
	loss_data_pixel = Loss_pixel
	loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_pixel_epoch_" + str(
		args.epochs) + "_" + str(
		time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
	scio.savemat(loss_filename_path, {'final_loss_pixel': loss_data_pixel})
	loss_data_ssim = Loss_ssim
	loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_ssim_epoch_" + str(
		args.epochs) + "_" + str(
		time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
	scio.savemat(loss_filename_path, {'final_loss_ssim': loss_data_ssim})
	# SSIM loss
	loss_data = Loss_all
	loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_all_epoch_" + str(
		args.epochs) + "_" + str(
		time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
	scio.savemat(loss_filename_path, {'final_loss_all': loss_data})
	# save model
	nest_model.eval()
	nest_model.cpu()
	save_model_filename = args.ssim_path[i] + '/' "Final_epoch_" + str(args.epochs) + "_" + \
						  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".model"
	save_model_path = os.path.join(args.save_model_dir_autoencoder, save_model_filename)
	torch.save(nest_model.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)

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

def check_paths(args):
	try:
		if not os.path.exists(args.vgg_model_dir):
			os.makedirs(args.vgg_model_dir)           #创建文件夹
		if not os.path.exists(args.save_model_dir):
			os.makedirs(args.save_model_dir)
	except OSError as e:
		print(e)
		sys.exit(1)


if __name__ == "__main__":
	main()
