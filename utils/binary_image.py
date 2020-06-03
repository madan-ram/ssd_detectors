# Python programe to illustrate  
# simple thresholding type on an image 
	  
# organizing imports  
import cv2  
import numpy as np  
from glob import glob
import random
import math
from scipy import ndimage

def _fill_image(gray):
	des = cv2.bitwise_not(gray)
	contour, hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

	cv2.drawContours(des, contour, -1, 255, 3)

	gray = cv2.bitwise_not(des)
	return gray

def compute_skew(image):

	filled_img = _fill_image(image)
	image_binary = cv2.bitwise_not(filled_img)
	height, width = image_binary.shape

	edges = cv2.Canny(image_binary,100,200)

	lines = cv2.HoughLinesP(edges, 1, math.pi/180, 100, minLineLength=width / 4.0, maxLineGap=20)
	lines = lines.reshape((-1, 4))

	angle = np.median(np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0]))

	return math.degrees(angle)

if __name__ == '__main__':

	from datetime import datetime

	# path to input image is specified and   
	# image is loaded with imread command
	# data_dir_path = "/Users/madanram/SoulOfCoder/field_extractor/sample_data/all_in_one/*.jpg"
	data_dir_path = "/Users/madanram/SoulOfCoder/field_extractor/sample_data/dummy/*.jpg"

	list_of_images = glob(data_dir_path)

	random.seed(datetime.now(), version=2)
	img_path = random.choice(list_of_images)

	image = cv2.imread(img_path)  
	
	# Randomly rotate image
	rand_angle = random.choice(range(0, 90, 10))
	print('Randomly rotate images', rand_angle)
	image = ndimage.rotate(image, -rand_angle)
	print(image.shape)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
	
	ret, thresh = cv2.threshold(gray_image, 120, 255, cv2.THRESH_TOZERO)
	angel = compute_skew(thresh)
	print('Image rotated at angle->', angel)
	deskewed_image = ndimage.rotate(thresh, angel)
		
	cv2.imwrite('test.jpg', deskewed_image)
	# Fix image if roted fully in large angel
	cv2.imshow('Binary Threshold', deskewed_image) 

		
	# De-allocate any associated memory usage   
	if cv2.waitKey(0) & 0xff == 27:  
		cv2.destroyAllWindows()
