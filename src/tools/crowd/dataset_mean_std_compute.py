import os

import cv2
import numpy as np

from tqdm import tqdm

image_dir = './images'

Bs = []
Gs = []
Rs = []
pixel_counts = []

for filename in tqdm(os.listdir(image_dir)):
  im = cv2.imread(os.path.join(image_dir, filename)) # BGR
  height, width, _ = im.shape

  Bs.append(np.sum(im[:, :, 0]) / 255)
  Gs.append(np.sum(im[:, :, 1]) / 255)
  Rs.append(np.sum(im[:, :, 2]) / 255)

  pixel_counts.append(height * width)

total_pixel_num = np.sum(pixel_counts)
B_mean = np.sum(Bs) / total_pixel_num
G_mean = np.sum(Gs) / total_pixel_num
R_mean = np.sum(Rs) / total_pixel_num
print('BGR mean is {}'.format(np.array([B_mean,G_mean,R_mean])))
'''
[0.42497516 0.44369858 0.46518538]
'''

B_squared_mean_diffs = []
G_squared_mean_diffs = []
R_squared_mean_diffs = []

for filename in tqdm(os.listdir(image_dir)):
  im = cv2.imread(os.path.join(image_dir, filename)) # BGR

  B_squared_mean_diffs.append(np.sum(np.square(im[:, :, 0] / 255 - B_mean)))
  G_squared_mean_diffs.append(np.sum(np.square(im[:, :, 1] / 255 - G_mean)))
  R_squared_mean_diffs.append(np.sum(np.square(im[:, :, 2] / 255 - R_mean)))

B_std = np.sqrt(np.sum(B_squared_mean_diffs) / total_pixel_num)
G_std = np.sqrt(np.sum(G_squared_mean_diffs) / total_pixel_num)
R_std = np.sqrt(np.sum(R_squared_mean_diffs) / total_pixel_num)
print('BGR std is {}'.format(np.array([B_std,G_std,R_std])))
'''
[0.29305324 0.28364128 0.28903846]
'''