import cv2
import os, sys
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import random
from glob import glob

def plt_rectangle(image, string, bb, color=(0, 255, 0), thickness=2):
	y_min, x_min, y_max, x_max=bb
	cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
	cv2.putText(image,	string, (x_min, y_min-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,  cv2.LINE_AA)
	return image

class Parser:

	def __init__(self, xmlfile, ignore_label_lst=["unlabelled"]):
		self.xmlfile = xmlfile
		tree = ET.parse(self.xmlfile, ET.XMLParser(encoding='utf-8'))
		self.root = tree.getroot()
		self.img_path = self.root.find('path').text
		self.img_width = self.root.find('size/width').text
		self.img_height = self.root.find('size/height').text
		self.ignore_label_lst = set(ignore_label_lst)

	def generator(self):

		img_depth = self.root.find('size/depth')
		for _object in self.root.findall('object'):

			label_name = _object.find("name").text

			if label_name in self.ignore_label_lst:
				continue
				
			xmin = _object.find("bndbox/xmin").text
			ymin = _object.find("bndbox/ymin").text
			xmax = _object.find("bndbox/xmax").text
			ymax = _object.find("bndbox/ymax").text
			text = _object.find("attributes/attribute/value")

			if text is None:
				text = ""
			else:
				text = text.text

			yield(label_name, int(float(xmin)), int(float(ymin)), int(float(xmax)), int(float(ymax)), text, int(float(self.img_width)), int(float(self.img_height)))

if __name__ == '__main__':

	dir_img_path = sys.argv[1]
	dir_xml_path = sys.argv[2]
	img_file_list = glob(dir_img_path+'/*.jpg')

	image_file_path = random.choice(img_file_list)
	img_file_name = image_file_path.split('/')[-1].split('.')[0]

	xml_file_path = os.path.join(dir_xml_path, img_file_name+'.xml')
	
	image =  cv2.imread(image_file_path)
	genr = Parser(xml_file_path)

	vis_img = image.copy()
	for data in genr.generator():
		label_name, xmin, ymin, xmax, ymax, text, img_width, img_height = data
		plt_rectangle(vis_img, label_name, [ymin, xmin, ymax, xmax])

	plt.imshow(vis_img)

	plt.show()
