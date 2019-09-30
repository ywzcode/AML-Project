import json
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from pycocotools.coco import maskUtils
def scale_analysis(coco:COCO):
    abs_areas = []
    relative_areas = []
    for ann_id,ann in coco.anns.items():
        if ann.get('ignore')==1:
            continue
        img_id = ann['image_id']
        area_img = coco.imgs[img_id]['width']*coco.imgs[img_id]['height']
        if ann['area']>area_img: # some error anno's size
            continue
        relative_areas.append(ann['area']/area_img)
        abs_areas.append(ann['area'])
    return abs_areas,relative_areas

def density_find(coco:COCO):
    num_object_to_img = {}
    num_object_range = [ [1,2], [2,5],[5,10],[10,50],[50,100],[100,100000] ]
    for rang in num_object_range:
        num_object_to_img[(rang[0],rang[1])] = []
    for img_id,x in coco.imgToAnns.items():
        num_object = len(x)
        for key in num_object_to_img.keys():
            if key[0]<=num_object<key[1]:
                num_object_to_img[key].append(img_id)
    return num_object_to_img

def occlusion_analysis(coco:COCO,iou_thresh):
    all_object_threshed = np.zeros(len(iou_thresh))
    for img_id,img_anns in coco.imgToAnns.items():
        bboxes = [ann['bbox'] for ann in img_anns]
        ious = maskUtils.iou(bboxes, bboxes, [0, ] * len(bboxes))
        ious[np.diag_indices_from(ious)] = 0
        ious = np.max(ious,axis=0)
        object_threshed = np.array([np.sum((ious >= thresh)) for thresh in iou_thresh])
        all_object_threshed += object_threshed
    all_object_threshed /= (len(coco.imgs))
    return all_object_threshed

names = ['crowd_train',
         'pascal_trainval0712',
         'coco_train2017',
         ]
datasets = {}
for file in names:
    dataset = COCO(os.path.join('./annotations',file+'.json'))
    abs_areas, relative_areas = scale_analysis(dataset)
    datasets[file] = [dataset,abs_areas,relative_areas]


## scale
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (12.0, 4.0)
plt.figure()
for dataset_name,dataset in datasets.items():
    name = dataset_name.split('_')[0]
    plt.hist(dataset[-1], bins=1000, cumulative=True, density=True, histtype='step',label=name)
plt.title('Fraction of object in the dataset vs scale of object\'s relative to the image')
plt.xlabel('Relative Scale') # (bbox area)/(image area)
plt.ylabel('CDF(Scale)')
plt.xticks(np.arange(10)/10)
plt.yticks(np.arange(10)/10)
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)
plt.grid(True)
plt.legend(loc='upper right',prop={'size': 8})
plt.savefig('relative_size.png')

## density
density_thresh = [1,2,3,5,10,20,30]
density_table = []
for dataset_name,dataset in datasets.items():
    dataset = dataset[0]
    num_object_all = np.array([len(x) for x in dataset.imgToAnns.values()])
    num_threshed = np.array([np.sum((num_object_all>=thresh)) for thresh in density_thresh])
    num_relative_threshed = num_threshed/len(dataset.imgs)
    density_table.append(num_threshed)
    density_table.append((num_relative_threshed)*100)
density_table = np.array(density_table).T
np.set_printoptions(precision=4,suppress=True)

print(names)
print(density_table)


## occlusion analysis
iou_thresh = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
occlusion_table = []
for dataset_name,dataset in datasets.items():
    dataset = dataset[0]
    all_object_threshed = occlusion_analysis(dataset,iou_thresh)
    occlusion_table.append(all_object_threshed)

occlusion_table = np.array(occlusion_table).T
np.set_printoptions(precision=4,suppress=True)
print(names)
print(occlusion_table)
