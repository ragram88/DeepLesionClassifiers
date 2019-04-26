'''
    name: DLProcessing.py
    purpose: Process DL_info.csv file using Pandas library
    date: April 13, 2019
'''

import pandas as pd
import numpy as np

if __name__ == '__main__':
	## Read in CSV
	df = pd.read_csv("../Data/DL_info.csv")
	df.info()
	## Extract bounding box information
	# df["Bounding_boxes"].head()
	df[["bb_xmin", "bb_ymin", "bb_xmax", "bb_ymax"]] = df["Bounding_boxes"].astype(str).str.split(",", expand=True).astype(float)
	df[["bb_xmin", "bb_ymin"]] = df[["bb_xmin", "bb_ymin"]].apply(np.floor)
	df[["bb_xmax", "bb_ymax"]] = df[["bb_xmax", "bb_ymax"]].apply(np.ceil)
	df["bb_xlength"] = (df.bb_xmax - df.bb_xmin)
	df["bb_ylength"] = (df.bb_ymax - df.bb_ymin)
	## Fix image file name
	df["File_name"] = "Images_png/"+df["File_name"].str[:-8]+"/"+df["File_name"].str[-7:] ## Zip files are in Image/.../<Image file name>
	# df["File_name"].head()
	## Format patient index, study index, series id, key slice index
	df["Patient_index"] = df["Patient_index"].astype(str).str.zfill(6)
	## Format study index
	df["Study_index"] = df["Study_index"].astype(str).str.zfill(2)
	## Format series id
	df["Series_ID"] = df["Series_ID"].astype(str).str.zfill(2)
	## Format key slice index
	df["Key_slice_index"] = df["Key_slice_index"].astype(str).str.zfill(3)
	## Filter out liver information
	lesion_type = 4 ## Liver is coarse lesion type 4
	max_length = 32 ## Maximum length
	max_width = 32 ## Maximum width
	liver = df[df["Coarse_lesion_type"]==lesion_type]
	liver = liver[liver["bb_xlength"]<=max_length]
	liver = liver[liver["bb_ylength"]<=max_width]
	## Write to CSV
	liver.to_csv(path_or_buf="../Data/DL_liver_32_32.csv", 
             index=False, 
             columns=["File_name",
					"Patient_index",
					"Study_index",
					"Series_ID",
					"Key_slice_index",
         			"bb_xmin",
         			"bb_ymin",
         			"bb_xmax",
         			"bb_ymax",
         			"bb_xlength",
         			"bb_ylength"])