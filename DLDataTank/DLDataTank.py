##
## name: DLDataTank.py
## purpose: Give users the ability to collect DL Lesion data
## date: 04/20/2019
##
##

import pandas as pd
import numpy as np
import urllib2
from zipfile import ZipFile
from StringIO import StringIO
import os
import random
import matplotlib.image as mpimg

class DLDataTank:
	# init - Initialize data tank
	# @param csv - CSV file with Data Lesion information
	# CSV must contain File_name, Patient_index, Study_index, Series_ID, Key_slice_index, bb_xmin, bb_xmax, bb_ymin, bb_ymax
	def __init__(self, csv):
		self.dtype = {"Patient_index": np.string_, 
			"Study_index": np.string_,
			"Series_ID": np.string_,
			"Key_slice_index": np.string_}
		self.csv = pd.read_csv(csv, dtype=self.dtype)
		## Attach Available boolean column for searching through available images easily
		self.csv["Available"] = False
		## Attach Local_file_name column for storing local file information
		self.csv["Local_file_name"] = self.csv["File_name"]
		## Portion of lesion array that has to have lesion pixels
		self.portion = 0.5
	# load - Load image files from remote url locations
	# @param url - string, url location
	# @param outputDir - string, output directory
	def load(self, urls, outputDir):
		print("Load data from links, unzip to output directories.")
		try:
			## Look through list of urls
			for u in urls:
				## Get absolute location of absolute
				absDir = os.path.abspath(outputDir)
				## Read zip file from URL
				response = urllib2.urlopen(u).read()
				zf = ZipFile(StringIO(response))
				## Find intersection of filenames
				intersect = list(set(zf.namelist()).intersection(self.csv["File_name"]))
				## Extract files and update values in csv table
				for f in intersect:
					zf.extract(f, absDir)
					self.csv.ix[self.csv["File_name"]==f,"Available"] = True
					self.csv.ix[self.csv["File_name"]==f,"Local_file_name"] = os.path.join(absDir, f)
					print("Unzipped "+f+", stored in "+os.path.join(absDir, f))
		except:
			raise
	# retrieve - Retrieve image files from local directory
	# @param images - list, string paths to image files
	def retrieve(self, images):
		print("Retrieve local data.")
		try:
			## Look through image array
			## Find entries in csv table that image array entries correspond with
			## Update csv table
			for i in images:
				patient_index = [n for n, s in enumerate(self.csv["Patient_index"]) if i.find(s)!=-1]
				study_index = set(patient_index).intersection([n for n, s in enumerate(self.csv["Study_index"]) if i.find(s)!=-1])
				series_index = set(study_index).intersection([n for n, s in enumerate(self.csv["Series_ID"]) if i.find(s)!=-1])
				key_slice_index = set(series_index).intersection([n for n, s in enumerate(self.csv["Key_slice_index"]) if i.find(s)!=-1])
				if(len(key_slice_index)==1):
					index = key_slice_index.pop()
					self.csv.ix[index,"Available"] = True
					self.csv.ix[index,"Local_file_name"] = os.path.abspath(i)
					print("Retrieved "+i)
		except:
			raise
	## generate - Generate length * width sized pixel arrays from images
	## @param length - int, length of pixel array
	## @param width - int, width of pixel array
	## @param number - int, number of images
	## @return list - [0.0(Non-lesion)/1.0(Lesion), [length * width pixel array]]
	def generate(self, length, width, number):
		print("Generate.")
		available = self.csv[self.csv["Available"]==True]
		try:
			if(len(available)>0):
				## Get random images
				files = random.sample(available["File_name"].tolist(), number)
				dataset = []
				# print(type(available["File_name"].tolist()))
				for f in files:
					img = available[available["File_name"]==f]
					print("Image generated from "+img.Local_file_name.item())
					## Get image pixels
					# print(type(img.Local_file_name.item()))
					image_pixels = mpimg.imread(img.Local_file_name.item())
					## Get classification
					## True - Has lesion
					## False - Does not have lesion
					## classification = True
					## classification = False
					classification = np.random.choice([True, False], 
						p=[0.5, 0.5])
					## Choose left-top corner point of array boundaries
					print("Bounding box location (xmin, xmax, ymin, ymax): ")
					print([img.bb_xmin.item(), 
						img.bb_xmax.item(), 
						img.bb_ymin.item(), 
						img.bb_ymax.item()])
					## Get array of values
					if classification == True:
						print("Generating lesion array.")
						x_min = max(0, int(img["bb_xmin"]-(1-self.portion)*length))
						x_max = min(512-length, int(img["bb_xmax"]-(self.portion*length)))
						y_min = max(0, int(img["bb_ymin"]-(1-self.portion)*width))
						y_max = min(512-width, int(img["bb_ymax"]-(self.portion*width)))
						x_left_top_corner = random.randint(x_min, x_max)
						y_left_top_corner = random.randint(y_min, y_max)
						print("Generated array location (xmin, xmax, ymin, ymax): ")
						print([x_left_top_corner,  
							x_left_top_corner+length-1, 
							y_left_top_corner, 
							y_left_top_corner+width-1])
						dataset.append([1.0,
								np.ravel(image_pixels[x_left_top_corner:x_left_top_corner+length,
											y_left_top_corner:y_left_top_corner+width]).tolist()])
					else:
						print("Generating non-lesion array.")
						## Find distance between edges of image and bounding box
						top_frame = img["bb_ymin"]
						bottom_frame = 512-img["bb_ymax"]
						left_frame = img["bb_xmin"]
						right_frame = 512-img["bb_xmax"]
						options = []
						## Add possible left top corner x, y options
						if (top_frame >= width).bool():
							options.append([[0, 512-length],[0, int(img["bb_ymin"]-width)]])
						if (bottom_frame >= width).bool():
							options.append([[0, 512-length],[int(img["bb_ymax"]+1), 512-width]])
						if (left_frame >= length).bool():
							options.append([[0, int(img["bb_xmin"]-length)],[0, 512-width]])
						if (right_frame >= length).bool():
							options.append([[int(img["bb_xmax"]+1), 512-length],[0, 512-width]])
						## Randomly select between options
						option = random.choice(options)
						x_left_top_corner = random.randint(option[0][0], option[0][1])
						y_left_top_corner = random.randint(option[1][0], option[1][1])
						print("Generated array location (xmin, xmax, ymin, ymax): ")
						print([x_left_top_corner,  
							x_left_top_corner+length-1, 
							y_left_top_corner, 
							y_left_top_corner+width-1])
						dataset.append([0.0,
								np.ravel(image_pixels[x_left_top_corner:x_left_top_corner+length,
											y_left_top_corner:y_left_top_corner+width]).tolist()])
				return dataset
		except:
			raise
	# getImages - Get list of images that are available in Data Tank
	# @return - list, string paths of all images available in Data Tank
	def getImages(self):
		print("Get Images.")
		return self.csv[self.csv["Available"]==True]