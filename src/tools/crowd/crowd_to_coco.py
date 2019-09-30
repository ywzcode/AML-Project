from PIL import Image
import json
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

category_id_dict = {'person': 1, 'mask': 2}
data_foler = r'./'


# TODO: vis
def vis(image_id, dict_crowd):
    img = Image.open(os.path.join(data_foler, 'Images', image_id + '.jpg'))
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(img)
    for box in dict_crowd['gtboxes']:
        if box['tag'] == 'mask':
            xmin, ymin, w, h = box['vbox']
            rect = patches.Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
        if box['tag'] == 'person':
            xmin, ymin, w, h = box['vbox']
            rect = patches.Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor='y', facecolor='none')
            ax.add_patch(rect)
    plt.show()
    return

def crowd_to_coco_json(crod_odgt_file: str, bbox_type='vbox'):
    subset = crod_odgt_file.split('_')[1].split('.')[0]
    json_dict = {"images": [], "annotations": [], "categories": []}
    json_dict['categories'].append({'supercategory': 'none', 'id': 1, 'name': 'person'})

    anno_id = 1
    image_id = 1
    max_person_per_img = 0
    num_all_person = 0
    num_all_person_ignored = 0
    num_all_mask = 0

    with open(os.path.join(data_foler, "annotations", crod_odgt_file), 'r') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        record = json.loads(line)
        img_name = record['ID'] + '.jpg'
        if os.path.exists(os.path.join(data_foler, 'images', img_name)):
            img = Image.open(os.path.join(data_foler, 'images', img_name))
        else:
            print('image not find:{}'.format(record['ID']))

        num_person = 0
        for box in record['gtboxes']:
            bbox = box[bbox_type]
            ignore = 0
            if "ignore" in box['head_attr']:
                ignore = box['head_attr']['ignore']
            if "ignore" in box['extra']:
                ignore = box['extra']['ignore']

            ## only consider person
            if box['tag'] =='person':
                anno_dict = {
                    "segmentation": [],
                    "area": bbox[2] * bbox[3],
                    "iscrowd": ignore,
                    "image_id": image_id,
                    "bbox": bbox,
                    "category_id": category_id_dict[box['tag']],
                    "id": anno_id,
                    "ignore": ignore,
                }
                json_dict['annotations'].append(anno_dict)
                anno_id += 1
            if box['tag'] == 'person':
                if ignore:
                    num_all_person_ignored += 1
                else:
                    num_all_person += 1
                    num_person += 1
            else:
                num_all_mask += 1
                if ignore != 1:
                    print('mask should be ignored in {}'.format(record['ID']))

        json_dict['images'].append({'file_name': img_name, 'height': img.size[1], 'width': img.size[0],
                                        'id': image_id})
        image_id += 1
        if num_person > max_person_per_img:
            max_person_per_img = num_person
    ## summary
    print("For {} subset".format(subset))
    print("There are max {} person per image".format(max_person_per_img))
    print("There are {} person within the dataset, {} are ignored and {} are not".format(num_all_person,num_all_person_ignored,num_all_person-num_all_person_ignored))
    print("There are {} ignored mask region within the dataset".format(num_all_mask))

    with open(os.path.join(data_foler,'annotations','crowd_'+subset+'.json'),'w') as f:
        json_str = json.dumps(json_dict)
        f.write(json_str)
        f.close()

    return json_dict

train_json = crowd_to_coco_json('annotation_train.odgt')
val_json = crowd_to_coco_json('annotation_val.odgt')