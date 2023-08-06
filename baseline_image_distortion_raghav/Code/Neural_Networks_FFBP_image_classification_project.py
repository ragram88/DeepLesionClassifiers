# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 23:17:11 2019
Neural Networks Class Project
@author: ramacr1
"""

import os
import pandas as pd
import random
import numpy as np
from medpy.io import load
from PIL import Image
from scipy import ndimage

def liver_lesions(images_in_DL,sub_x,sub_y):
    #identify liver lesions (Coarse_lesion_type=4) 
    is_liver = images_in_DL['Coarse_lesion_type']==4
    liver_sel_lesions = images_in_DL[(is_liver)]
    liver_sel_lesions.index=range(0,liver_sel_lesions.shape[0])
    
    x1 = [None] * liver_sel_lesions.shape[0]
    y1 = [None] * liver_sel_lesions.shape[0]
    x2 = [None] * liver_sel_lesions.shape[0]
    y2 = [None] * liver_sel_lesions.shape[0]
    xbound = [None] * liver_sel_lesions.shape[0]
    ybound = [None] * liver_sel_lesions.shape[0]
    is_bounded = [None] * liver_sel_lesions.shape[0]
    #calculate size of bounding box for lesions
    for ix in range(0,liver_sel_lesions.shape[0]):
        bound_rec = liver_sel_lesions['Bounding_boxes'][ix]
        bounded_box = [x.strip() for x in bound_rec.split(',')]
        x1[ix] = float(bounded_box[0])
        y1[ix] = float(bounded_box[1])
        x2[ix] = float(bounded_box[2])
        y2[ix] = float(bounded_box[3])
        xbound[ix] = x2[ix]  - x1[ix]
        ybound[ix] = y2[ix] - y1[ix]
        is_bounded[ix] = (xbound[ix]<=sub_x) and (ybound[ix]<=sub_y)
   
    #calculate dimensions of bounding box
    liver_sel_lesions.loc[liver_sel_lesions.index.tolist(), 'bounded_x1'] = x1
    liver_sel_lesions.loc[liver_sel_lesions.index.tolist(), 'bounded_y1'] = y1
    liver_sel_lesions.loc[liver_sel_lesions.index.tolist(), 'bounded_x2'] = x2
    liver_sel_lesions.loc[liver_sel_lesions.index.tolist(), 'bounded_y2'] = y2

    liver_sel_lesions.loc[liver_sel_lesions.index.tolist(), 'x_size'] = np.subtract(np.ceil(x2),np.floor(x1))
    liver_sel_lesions.loc[liver_sel_lesions.index.tolist(), 'y_size'] = np.subtract(np.ceil(y2),np.floor(y1))
        
    #bounding box of lesion needs to be less than 32-by-32 pixels

    liver_bounded_lesions = liver_sel_lesions[is_bounded]
    return(liver_bounded_lesions)

def lesion_subregion(liver_bounded_lesions, sub_x, sub_y):
    #choose a 32-by-32 subregion with the lesion (bounded box) for each of the 20 selected images
    sub_x_lower = [None] * liver_bounded_lesions.shape[0]
    sub_y_lower = [None] * liver_bounded_lesions.shape[0]
    sub_x_upper = [None] * liver_bounded_lesions.shape[0]
    sub_y_upper = [None] * liver_bounded_lesions.shape[0]
    random.seed(12321)
    for ib in range(0,liver_bounded_lesions.shape[0]):
        x_extra = sub_x - liver_bounded_lesions.x_size[liver_bounded_lesions.index[ib]]        
        y_extra = sub_y - liver_bounded_lesions.y_size[liver_bounded_lesions.index[ib]]
        x_lb = random.randint(0,x_extra)
        x_ub = x_extra-x_lb
        y_lb = random.randint(0,y_extra)
        y_ub = y_extra-y_lb
        sub_x_lower[ib] = int(np.floor(liver_bounded_lesions.bounded_x1[liver_bounded_lesions.index[ib]]) - x_lb)
        sub_x_upper[ib] = int(np.ceil(liver_bounded_lesions.bounded_x2[liver_bounded_lesions.index[ib]]) + x_ub)
        sub_y_lower[ib] = int(np.floor(liver_bounded_lesions.bounded_y1[liver_bounded_lesions.index[ib]])- y_lb)
        sub_y_upper[ib] = int(np.ceil(liver_bounded_lesions.bounded_y2[liver_bounded_lesions.index[ib]]) + y_ub)

    return [sub_x_lower, sub_x_upper, sub_y_lower, sub_y_upper]

def nonlesion_subregion(liver_bounded_lesions,x1,x2,y1,y2, sub_x, sub_y):
    #choose a 32-by-32 subregion without the lesion for each of the 20 selected images
    sub_x_lower = [None] * liver_bounded_lesions.shape[0]
    sub_y_lower = [None] * liver_bounded_lesions.shape[0]
    sub_x_upper = [None] * liver_bounded_lesions.shape[0]
    sub_y_upper = [None] * liver_bounded_lesions.shape[0]
    random.seed(12321)
    for ib in range(0,liver_bounded_lesions.shape[0]):
        x_lb = random.randint(0,(512-(sub_x+1)))
        y_lb = random.randint(0,(512-(sub_y+1)))
        x_ub = x_lb + sub_x
        y_ub = y_lb + sub_y
        while(((x1[ib] <= x_lb <= x2[ib]) or (x1[ib] <= x_ub <= x2[ib])) and ((y1[ib] <= y_lb <= y2[ib]) or (y1[ib] <= y_ub <= y2[ib]))):
            x_lb = random.randint(0,(512-(sub_x+1)))
            y_lb = random.randint(0,(512-(sub_y+1)))
            x_ub = x_lb + sub_x
            y_ub = y_lb + sub_y
        sub_x_lower[ib] = x_lb
        sub_x_upper[ib] = x_ub
        sub_y_lower[ib] = y_lb 
        sub_y_upper[ib] = y_ub

    return [sub_x_lower, sub_x_upper, sub_y_lower, sub_y_upper]

def main():
    #randomly choose zipped folders from which the images in the image dataset will be selected
    random.seed(12321)
    zipfile_num = random.randint(1,56) #zip file: 36
    zipfile_num2 = random.randint(1,56) #zip file: 56
    zipfile_num3 = random.randint(1,56) #zip file: 48
    zipfile_num4 = random.randint(1,56) #zip file: 16

    os.chdir('C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\')
    lesion_info = pd.read_csv("DL_info.csv") #metadata about images and lesions
    
    #derive file path for images in each of the randomly chosen zip-files
    path='C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\Images_png_36\\Images_png\\'
    samp_images = []
    for r,d,f in os.walk(path): #r-root, d-directories, f-files
        for file in f:
            samp_images.append(os.path.basename(r) + "_" + file)

    path='C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\Images_png_56\\Images_png\\'
    samp_images2 = []
    for r,d,f in os.walk(path): #r-root, d-directories, f-files
        for file in f:
            samp_images2.append(os.path.basename(r) + "_" + file)

    path='C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\Images_png_48\\Images_png\\'
    samp_images3 = []
    for r,d,f in os.walk(path): #r-root, d-directories, f-files
        for file in f:
            samp_images3.append(os.path.basename(r) + "_" + file)

    path='C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\Images_png_16\\Images_png\\'
    samp_images4 = []
    for r,d,f in os.walk(path): #r-root, d-directories, f-files
        for file in f:
            samp_images4.append(os.path.basename(r) + "_" + file)
   
    #identify images from all randomly chosen zip-files     
    samp_images_all = samp_images + samp_images2 + samp_images3 + samp_images4
    images_in_DL = lesion_info[lesion_info['File_name'].isin(samp_images_all)]
    liver_bounded_lesions = liver_lesions(images_in_DL,32,32) #identify names and other details of images with liver lesions and a bounded box of size less than 32-by-32 pixels
    liver_bounded_lesions.to_csv("NNProject_liver_image_dataset_info_050319.csv")        
    
    random.seed(12321)
    samp_size = liver_bounded_lesions.shape[0]
    sel_images = liver_bounded_lesions.index[random.sample(range(samp_size),21)]
    liver_bounded_lesions_21_imgs = liver_bounded_lesions[liver_bounded_lesions.index.isin(sel_images)]
    #Obtain boundaries of 32-by-32 subregions from images: one containing the lesion and the other without it
    [sub_x_lower, sub_x_upper, sub_y_lower, sub_y_upper] = lesion_subregion(liver_bounded_lesions_21_imgs,32,32)
    [sub2_x_lower, sub2_x_upper, sub2_y_lower, sub2_y_upper] = nonlesion_subregion(liver_bounded_lesions_21_imgs, sub_x_lower, sub_x_upper, sub_y_lower, sub_y_upper,32,32)
    liver_bounded_lesions_21_imgs.to_csv("NNProject_liver_image_dataset_info_N21.csv")        
        
    #save subregion boundaries for each image to a .csv file
    subregion_boundaries = pd.DataFrame(liver_bounded_lesions_21_imgs.File_name)
    subregion_boundaries['lesion_x_lower'] = sub_x_lower
    subregion_boundaries['lesion_x_upper'] = sub_x_upper
    subregion_boundaries['lesion_y_lower'] = sub_y_lower
    subregion_boundaries['lesion_y_upper'] = sub_y_upper
    subregion_boundaries['nonlesion_x_lower'] = sub2_x_lower
    subregion_boundaries['nonlesion_x_upper'] = sub2_x_upper
    subregion_boundaries['nonlesion_y_lower'] = sub2_y_lower
    subregion_boundaries['nonlesion_y_upper'] = sub2_y_upper
    
    subregion_boundaries.to_csv("Subregion_boundaries_050319.csv")
    subregion_boundaries.to_csv("Subregion_boundaries_N21.csv")

    num_images = liver_bounded_lesions_21_imgs.shape[0]
    img = np.zeros((512,512,num_images))
    sub32_les = np.zeros((32,32,num_images))
    sub32_nonles = np.zeros((32,32,num_images))
    img_header = [None] * num_images
    
    #Adding Gaussian/Gamma noise to images
    random.seed(12321)
    mean = 1
    var = 1
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (32, 32, num_images))
    
    random.seed(12321)
    shape, scale = 1, 1 #mean = 2*5 = 10, variance = 2*5*5 = 50
    gamma = np.random.gamma(shape, scale, (32,32,num_images))
    
    #read images and 32-by-32 subregions given the names of the images obtained above
    #Use answer to question #3 in FAQ.pdf to correctly process the images
    os.chdir('C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\Project_img_dataset\\all_images')
    for indx in range(0,num_images):
        img[:,:,indx], img_header[indx] = load(str(liver_bounded_lesions_21_imgs.File_name[liver_bounded_lesions_21_imgs.index[indx]]))  #3-by-512-512
        img[:,:,indx] = np.array(img[:,:,indx], dtype=np.int32)
        img[:,:,indx] = img[:,:,indx] - 32768
        for i in range(0,511):
            for j in range(0,511):
                img[i,j,indx] = min(255, max(0, (img[i,j,indx]-(-175))/(275-(-175))*255))
        sub32_les[:,:,indx] = img[sub_x_lower[indx]:sub_x_upper[indx],sub_y_lower[indx]:sub_y_upper[indx],indx]
        sub32_nonles[:,:,indx] = img[sub2_x_lower[indx]:sub2_x_upper[indx],sub2_y_lower[indx]:sub2_y_upper[indx],indx]
    
    Gss_noisy32_les = sub32_les + gaussian
    Gss_noisy32_nonles = sub32_nonles + gaussian
    Gamma_noisy32_les = sub32_les + gamma
    Gamma_noisy32_nonles = sub32_nonles + gamma
    transform_les = ndimage.rotate(sub32_les, 180, reshape=False)
    transform_nonles = ndimage.rotate(sub32_nonles, 180, reshape=False) 

    #save original and distorted images with/without lesions
    os.chdir('C:\\Users\\ramacr1\\Desktop\\Desktop\\DEng\\SP19_Neural_Networks\\DeepLesionClassifiers\\Sub_images\\')
    for imgnum in range(0,num_images):
         origimg_les = Image.fromarray(np.array(sub32_les[:,:,imgnum], dtype=np.int8), mode="L")
         origimg_nonles = Image.fromarray(np.array(sub32_nonles[:,:,imgnum], dtype=np.int8), mode="L")
         Gssimg_les = Image.fromarray(np.array(Gss_noisy32_les[:,:,imgnum], dtype=np.int8), mode="L")
         Gssimg_nonles = Image.fromarray(np.array(Gss_noisy32_nonles[:,:,imgnum], dtype=np.int8), mode="L")
         Gammaimg_les = Image.fromarray(np.array(Gamma_noisy32_les[:,:,imgnum], dtype=np.int8), mode="L")
         Gammaimg_nonles = Image.fromarray(np.array(Gamma_noisy32_nonles[:,:,imgnum], dtype=np.int8), mode="L")
         rot45img_les = Image.fromarray(np.array(transform_les[:,:,imgnum], dtype=np.int8), mode="L")
         rot45img_nonles = Image.fromarray(np.array(transform_nonles[:,:,imgnum], dtype=np.int8), mode="L")
         origimg_les.save('Original\\lesion\\Original_lesion_sub_img_' + str(imgnum) + '.png')      
         origimg_nonles.save('Original\\nonlesion\\Original_nonlesion_sub_img_' + str(imgnum) + '.png')      
         Gssimg_les.save('Gaussian_noise\\lesion\\GaussNoise_lesion_sub_img_' + str(imgnum) + '.png')      
         Gssimg_nonles.save('Gaussian_noise\\nonlesion\\GaussNoise_nonlesion_sub_img_' + str(imgnum) + '.png')      
         Gammaimg_les.save('Gamma_noise\\lesion\\GammaNoise_lesion_sub_img_' + str(imgnum) + '.png')      
         Gammaimg_nonles.save('Gamma_noise\\nonlesion\\GammaNoise_nonlesion_sub_img_' + str(imgnum) + '.png')      
         rot45img_les.save('Rotate180\\lesion\\Rotate_lesion_sub_img_' + str(imgnum) + '.png')      
         rot45img_nonles.save('Rotate180\\nonlesion\\Rotate_nonlesion_sub_img_' + str(imgnum) + '.png')      
       
    orig_ds = []
    GaussNoise_ds = []
    GammaNoise_ds = []
    rotate180_ds = []
    img_nums = np.arange((2*num_images))
    np.random.seed(12321)
    np.random.shuffle(img_nums)
    
    for sno in range(0,(2*num_images)):
        img_index = img_nums[sno]
        if img_index < (num_images):
            orig_ds.append([0.0, np.ravel(sub32_nonles[:,:,img_index].tolist()),liver_bounded_lesions_21_imgs['File_name']])
            GaussNoise_ds.append([0.0, np.ravel(Gss_noisy32_nonles[:,:,img_index].tolist()),liver_bounded_lesions_21_imgs['File_name']])
            GammaNoise_ds.append([0.0, np.ravel(Gamma_noisy32_nonles[:,:,img_index].tolist()),liver_bounded_lesions_21_imgs['File_name']])
            rotate180_ds.append([0.0, np.ravel(transform_nonles[:,:,img_index].tolist()),liver_bounded_lesions_21_imgs['File_name']])
        else:
            orig_ds.append([1.0, np.ravel(sub32_les[:,:,(img_index-num_images)].tolist()),liver_bounded_lesions_21_imgs['File_name']])
            GaussNoise_ds.append([1.0, np.ravel(Gss_noisy32_les[:,:,(img_index-num_images)].tolist()),liver_bounded_lesions_21_imgs['File_name']])
            GammaNoise_ds.append([1.0, np.ravel(Gamma_noisy32_les[:,:,(img_index-num_images)].tolist()),liver_bounded_lesions_21_imgs['File_name']])
            rotate180_ds.append([1.0, np.ravel(transform_les[:,:,(img_index-num_images)].tolist()),liver_bounded_lesions_21_imgs['File_name']])
    
    #original, Gaussian Noise and Gamma Noise images
    full_ds = []
    full_ds = orig_ds + GaussNoise_ds + GammaNoise_ds

    #original, Gaussian Noise and Gamma Noise images (randomized order)
    np.random.seed(12321)
    allimg_nums = np.arange(len(full_ds))
    np.random.shuffle(allimg_nums)
    mixedimg_nums = allimg_nums[0:(2*num_images)]
    mixed_ds = np.array(full_ds)[mixedimg_nums,:].tolist()

    #original, Gaussian Noise images (randomized order)
    Gssorig_ds = orig_ds + GaussNoise_ds
    Gnoise_img_nums = np.arange(len(Gssorig_ds))
    np.random.shuffle(Gnoise_img_nums)
    Gnoise_mixed_img_nums = Gnoise_img_nums[0:(2*num_images)]
    Gnoise_orig_ds = np.array(Gssorig_ds)[Gnoise_mixed_img_nums,:].tolist()
    
    #original, Gamma Noise images (randomized order)
    Gammaorig_ds = orig_ds + GammaNoise_ds
    Gamma_img_nums = np.arange(len(Gammaorig_ds))
    np.random.shuffle(Gamma_img_nums)
    Gamma_mixed_img_nums = Gamma_img_nums[0:(2*num_images)]
    Gamma_orig_ds = np.array(Gammaorig_ds)[Gamma_mixed_img_nums,:].tolist()

    #original, rotated images (randomized order)
    rotate_orig_ds = orig_ds + rotate180_ds
    rot180_imgs= np.arange(len(rotate_orig_ds))
    np.random.shuffle(rot180_imgs)
    rot180_img_nums = rot180_imgs[0:(2*num_images)]
    rotate180_orig_ds = np.array(rotate_orig_ds)[rot180_img_nums,:].tolist()

if __name__=="__main__":
    main()