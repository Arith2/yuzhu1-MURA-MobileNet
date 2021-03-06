import os
import numpy as np
import cv2
import random
import keras.backend as K

def load_path(root_path, size = 512):
	'''
	load MURA data
	'''

	Path = []
	labels = []
	for root, dirs, files in os.walk(root_path): # read all images
		for name in files:
			path_1 = os.path.join(root, name)
			Path.append(path_1)
			if root.split('_')[-1]=='positive':	 # positive label 1，otherwise 0；
				labels+=[1]   	          	 
			else:
			    labels+=[0]

	labels = np.asarray(labels)
	return Path, labels

def load_image(Path = '/home/yu/Documents/tensorflow/MURA/MURA_densenet_v1/valid/XR_ELBOW', size = 512):
	# the input Path should be the absolute path of the image
	Images = []
	for path in Path:
		image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		image = cv2.resize(image, (size, size))
		image = randome_rotation_flip(image, size)
		Images.append(image)
		print(np.array(image).shape)

	Images = np.asarray(Images).astype('float32')

	mean = np.mean(Images[:, :, :])			#normalization
	std = np.std(Images[:, :, :])
	Images[:, :, :] = (Images[:, :, :] - mean) / std
	
	if K.image_data_format() == "channels_first":
		Images = np.expand_dims(Images,axis=1)		   # expand the first dimension 
	if K.image_data_format() == "channels_last":
		Images = np.expand_dims(Images,axis=3)             
	return Images


def randome_rotation_flip(image,size = 512):
	if random.randint(0,1):
		iamge = cv2.flip(image,1) 

	if random.randint(0,1):
		angle = random.randint(-30,30)
		M = cv2.getRotationMatrix2D((size/2,size/2),angle,1)
		image = cv2.warpAffine(image,M,(size,size))
	return image



if __name__ == '__main__':
	a, b = load_path(root_path ='/home/yu/Documents/tensorflow/MURA/MURA-v1.1/train', size = 320)
	print(a)
	print(b)
#	load_image([],[])
