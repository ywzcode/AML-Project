#import _init_paths
import numpy as np
import scipy.io as sio
import os, sys, cv2
import argparse
from pycocotools.cocoeval import COCOeval
import copy
import matplotlib.pyplot as plt
import pycocotools.coco as coco


CAT_TO_NAMES = {0:'Human',1:'Background'}
COLORS = [((np.random.random((3, )) * 0.6 + 0.4)*255).astype(np.uint8) for _ in range(1)]
parser = argparse.ArgumentParser(description='get args for vis')
parser.add_argument('--type', type=str, help='CenterNet or FasterRCNN or RepPoints')
args = parser.parse_args()

def _coco_box_to_bbox(box):
	'''
	Converting the box coordinate to 
	(left-up corner, right-bottom) format
	'''
	bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],dtype=np.int32)
	return bbox

def add_box(image, bbox, sc, cat_id,k):
    '''
    Add box to image
    On the left-up corner, indicate the human number and the score
    '''
    cat_id = cat_id - 1 
    cat_name = CAT_TO_NAMES[cat_id]
    txt = '{}_{} {:.02f}'.format(cat_name,k,sc)

    cat_size  = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    color = np.array(COLORS[cat_id]).astype(np.int32).tolist()
    
    if bbox[1] - cat_size[1] - 2 < 0:
        cv2.rectangle(image,
                      (bbox[0], bbox[1] + 2),
                      (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
                      color, -1)
        cv2.putText(image, txt,
                    (bbox[0], bbox[1] + cat_size[1] + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
    else:
        cv2.rectangle(image,
                      (bbox[0], bbox[1] - cat_size[1] - 2),
                      (bbox[0] + cat_size[0], bbox[1] - 2),
                      color, -1)
        cv2.putText(image, txt,
                    (bbox[0], bbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
    cv2.rectangle(image,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color, 2)
    return image
def process_single_img(coco_gt,coco_dets,IMG_PATH,img_id,save_path):
	'''
	Add bounding box for each image
	coco is the coco class (with gt json data)
	coco_dets: the prediction loaded in coco class
	IMG_PATH: path of images, here, we pick 4 images to vis
	img_id: id of the image
	save path: path to save
	'''

	img_info = coco_gt.loadImgs(ids=[img_id])[0]
	img_path = IMG_PATH + img_info['file_name']
	img = cv2.imread(img_path)
	pred_img = img.copy()
	pred_ids = coco_dets.getAnnIds(imgIds=[img_id])
	preds = coco_dets.loadAnns(pred_ids)
	k = 1
	for j, pred in enumerate(preds):
	    bbox = _coco_box_to_bbox(pred['bbox'])
	    sc = pred['score']
	    cat_id = pred['category_id']
	    if sc > 0.5:
	        pred_img = add_box(pred_img, bbox, sc, cat_id,k)
	        k = k+1

	cv2.imwrite(os.path.join(save_path,img_info['file_name']),pred_img)

def main():
	num_classes = 1
	if args.type == 'RepPoints':
		result_file_path = './results-reppoints/results.json'
		save_path = 'results_vis/RepPoints'
	elif args.type == 'CenterNet':
		result_file_path = './results-center/results.json'
		save_path = 'results_vis/CenterNet'
	elif args.type == 'FasterRCNN':
		result_file_path = './results-Faster/results.json'
		save_path = 'results_vis/FasterRCNN'
	if not os.path.exists(save_path):
		os.mkdir(save_path)

	img_names = []
	img_ids = []
	imgs_file = './imgs-to-vis/'
	for img_name in os.listdir(imgs_file):
		img_names.append(img_name)

	coco_gt = coco.COCO('./crowd_val.json')
	coco_dets = coco_gt.loadRes(result_file_path)

	for key in coco_gt.imgs:
	    if coco_gt.imgs[key]['file_name'] in img_names:
	        img_ids.append(coco_gt.imgs[key]['id'])

	for img_id in img_ids:
		process_single_img(coco_gt,coco_dets,imgs_file,img_id,save_path)

if __name__ == '__main__':
	main()







	
