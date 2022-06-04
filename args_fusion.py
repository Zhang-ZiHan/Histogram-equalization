
class args():
	# training args
	epochs = 2  #"number of training epochs, default is 2"
	batch_size = 1  #"batch size for training, default is 4"
	# dataset = "/data/Disk_B/KAIST-RGBIR/lwir"  # the dataset path in your computer
	dataset = "/data/Disk_B/MSCOCO2014/train2014/"
	# dataset = r'F:\database\MS-COCO2014\train2014'      #r用于转译
	HEIGHT = 256
	WIDTH = 256

	save_model_dir_autoencoder = "models/nestfuse_autoencoder"
	save_loss_dir = './models/loss_autoencoder/'

	cuda = 1      #是否使用cuda
	ssim_weight = [1,10,100,1000,10000]    #结构相似性损失函数权重
	ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']

	lr = 1e-4  #"learning rate, default is 0.001"
	lr_light = 1e-4  # "learning rate, default is 0.001"
	log_interval = 10  #"number of images after which the training loss is logged, default is 500"  记录训练损失后的图像数，默认为500
	resume = None         #

	# for test, model_default is the model used in paper
	# model_default = './models/nestfuse_1e2.model'
	# 原编码器
	# model_default ='models/nestfuse_autoencoder/1e2/Final_epoch_2_Sun_Jun__6_16_50_56_2021_1e2.model'
	# 6月10日 原编码器
	# model_default = 'models/nestfuse_autoencoder/1e2/Final_epoch_2_Thu_Jun_10_16_51_13_2021_1e2.model'
	# 现编码器无均衡化
	# model_default ='models/nestfuse_autoencoder/1e2/Final_epoch_2_Mon_Jun__7_21_26_34_2021_1e2.model'
	# 最终 RXD不减半
	# model_default = 'models/nestfuse_autoencoder/1e2/Final_epoch_2_Wed_Jun__9_19_37_56_2021_1e2.model'
	# 最终
	model_default ='models/nestfuse_autoencoder/1e2/Final_epoch_2_Sun_Jun__6_12_17_13_2021_1e2.model'
	# 最终 编码器不压缩 有均衡化 目前效果最好
	# model_default = 'models/nestfuse_autoencoder/1e2/Final_epoch_2_Wed_Jun__9_23_39_33_2021_1e2.model'
	#最终 编码器压缩 有均衡化 目前效果最好
	# model_default = 'models/nestfuse_autoencoder/1e2/Final_epoch_2_Thu_Jun_10_15_02_59_2021_1e2.model'
	# 原+resnet
	# model_default = 'models/nestfuse_autoencoder/1e2/Final_epoch_2_Thu_Jun_10_15_19_10_2021_1e2.model'
	# 原+densenet
	# model_default = 'models/nestfuse_autoencoder/1e2/Final_epoch_2_Thu_Jun_10_16_06_56_2021_1e2.model'
	# densenet + resnet
	# model_default = 'models/nestfuse_autoencoder/1e2/Final_epoch_2_Thu_Jun_10_16_17_46_2021_1e2.model'
	# 原 + densenet + resnet
	# model_default = 'models/nestfuse_autoencoder/1e2/Final_epoch_2_Thu_Jun_10_16_34_45_2021_1e2.model'
	# 添加relu 无均衡
	# model_default = 'models/nestfuse_autoencoder/1e2/Final_epoch_2_Thu_Jun_10_21_06_11_2021_1e2.model'
	# 添加relu 有均衡
	# model_default = 'models/nestfuse_autoencoder/1e2/Final_epoch_2_Thu_Jun_10_21_21_16_2021_1e2.model'
	model_deepsuper = './models/nestfuse_1e2_deep_super.model'


