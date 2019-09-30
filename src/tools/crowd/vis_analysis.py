from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
from detectors.base_detector import BaseDetector
from utils.image import get_affine_transform
from models.utils import flip_tensor
from utils.debugger import Debugger
from skimage.segmentation import slic
import skimage
class FeatureExtractor(torch.nn.Module):
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers='hm', output=True):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.target_layers = target_layers
        self.target_val = None
        self.output = output
        for name, m in self.model.named_modules():
            if name == self.target_layers:
                m.register_forward_hook(self.save_hook)

    def save_hook(self, module, input, output):
        self.target_val = output if self.output else input
        return

    def __call__(self, x):
        return self.model(x)


def pre_process(detector, image, scale,mask=None,return_resized_img=False):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width = int(width * scale)
    if detector.opt.fix_res:
        inp_height, inp_width = detector.opt.input_h, detector.opt.input_w
        c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
    else:
        inp_height = (new_height | detector.opt.pad) + 1
        inp_width = (new_width | detector.opt.pad) + 1
        c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
        s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
        resized_image, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR)
    if mask is not None:
        inp_image = inp_image * mask[:, :, np.newaxis]
    if return_resized_img:
        resized_img = inp_image.copy()
    inp_image = ((inp_image / 255. - detector.mean) / detector.std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    if detector.opt.flip_test:
        images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s,
            'out_height': inp_height // detector.opt.down_ratio,
            'out_width': inp_width // detector.opt.down_ratio}
    if not return_resized_img:
        return images,meta
    else:
        return images,meta,resized_img

def error_bound_saliency(opt, img_id, loc=None, error_bound=0.1):

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    Detector = detector_factory[opt.task]

    ### simply run the detector and save the objectness heat map and the detection results
    split = 'val' if not opt.trainval else 'test'
    dataset = Dataset(opt, split)
    detector = Detector(opt)
    # use the FeatureExtractor to regester the hook to get activation value
    # to find the name of target_layers, see the model.named_modules()
    feature_extractor = FeatureExtractor(detector.model,target_layers='hm')
    detector.model = feature_extractor
    feature_extractor.eval()
    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])
    detector.run(img_path)


    ### get saliency mask
    ### Not due to the input image is usually resized and padding, we get the mask on the resized image
    ### for error, we use L1 loss.


    ## gradually increase the rect center on the image coor untile the error is lower the boundry
    debug_dir = detector.opt.debug_dir
    scale = 1.0
    debugger = Debugger(dataset=detector.opt.dataset, ipynb=(detector.opt.debug==3),
                        theme=detector.opt.debugger_theme)

    image_org = cv2.imread(img_path)
    image, meta,resized_img = pre_process(detector,image_org, scale,mask=None,return_resized_img=True)
    _, _, h, w = image.size()
    down_sample_rate = h / feature_extractor.target_val.size(2)
    # get the loc[center_h,center_w] on the resized image and corresponding [fh,fw] on feature map
    if loc is None: # if loc [center_h,center_w] is not specified, use the location of the max value
        ind = torch.argmax(feature_extractor.target_val[0].sum(dim=0))
        fh = ind // feature_extractor.target_val.size(3)
        fw = ind % feature_extractor.target_val.size(3)
        center_h = fh * down_sample_rate
        center_w = fw * down_sample_rate
        val = feature_extractor.target_val[0, :, fh, fw]
        print([center_h,center_w])
    else:
        center_h, center_w = loc
        fh = int(center_h / down_sample_rate)
        fw = int(center_w / down_sample_rate)
        val = feature_extractor.target_val[0, :, fh, fw]

    loss_fn = lambda x: torch.mean(torch.pow((x - val), 2))
    area_increment = np.prod(image.size()) / 1000.0
    area = 0
    ratio = 1.0  # w/h = 1.0 increased rect ratio
    error = 1e10
    mask = np.zeros([h, w])  # [H,W]
    while (error > error_bound):
        print("it:{} error:{}".format(area//area_increment,error))
        area += area_increment
        bh = np.sqrt(area / ratio)
        bw = area / bh
        mask = np.zeros([h, w])
        hmin, hmax = max(int(center_h - bh / 2), 0), min(int(center_h + bh / 2) + 1, h - 1)
        wmin, wmax = max(int(center_w - bw / 2), 0), min(int(center_w + bw / 2) + 1, w - 1)
        mask[hmin:hmax, wmin:wmax] = 1
        image_masked,_ = pre_process(detector, image_org, 1.0, mask)
        image_masked = image_masked.to(opt.device)
        with torch.no_grad():
            feature_extractor(image_masked)
        error = loss_fn(feature_extractor.target_val[0, :, fh, fw])
    print("it:{} error:{}".format(area // area_increment, error))
    # draw the rect mask on resized_image and save
    rect_mask_img_save_name = 'rect_mask_{:.1f}'.format(scale)
    debugger.add_blend_img(resized_img, debugger.gen_colormap(mask[np.newaxis,:,:]), rect_mask_img_save_name)
    kernel_hmin,kernel_hmax = max(int(center_h - down_sample_rate / 2), 0), min(int(center_h + down_sample_rate / 2) + 1, h - 1)
    kernel_wmin, kernel_wmax = max(int(center_w - down_sample_rate / 2), 0), min(int(center_w + down_sample_rate / 2) + 1, w - 1)
    debugger.imgs[rect_mask_img_save_name][kernel_hmin:kernel_hmax, kernel_wmin:kernel_wmax] = [255,0,0]# green

    ## get saliency superpixel
    rect_img =  resized_img[hmin:hmax, wmin:wmax]
    segments = slic(rect_img,n_segments=30)#[hmin:hmax, wmin:wmax]
    un_removed_superpixel = list(np.unique(segments))
    rect_segment_mask = np.ones_like(segments)
    while(error<error_bound):
        # find superpixel whose removement leads to lowest error
        lowest_error = 1e10
        lowest_error_ind = -1
        for i in un_removed_superpixel:
            mask = np.zeros([h, w])
            mask[hmin:hmax, wmin:wmax] = rect_segment_mask*(segments!=i)
            image_masked,_ = pre_process(detector, image_org, 1.0, mask)
            image_masked = image_masked.to(opt.device)
            with torch.no_grad():
                feature_extractor(image_masked)
            cur_error = loss_fn(feature_extractor.target_val[0, :, fh, fw])
            if cur_error<lowest_error:
                lowest_error = cur_error
                lowest_error_ind = i
        if not lowest_error<error_bound:
            break
        else:
            un_removed_superpixel.remove(lowest_error_ind)
            error =lowest_error
            rect_segment_mask = rect_segment_mask*(segments!=lowest_error_ind)
            print("error={} remaining super pixel:{}".format(error,len(un_removed_superpixel)))

    # draw the segmentation saliency mask on resized_image and save
    mask = np.zeros([h, w])
    mask[hmin:hmax, wmin:wmax] = rect_segment_mask

    inp_image = resized_img*mask[:,:,np.newaxis].astype(np.uint8)
    debugger.add_img(inp_image,'masked_img')
    mask_img_save_name = 'mask_{:.1f}'.format(scale)
    debugger.add_blend_img(resized_img, debugger.gen_colormap(mask[np.newaxis, :, :]), mask_img_save_name)
    debugger.imgs[mask_img_save_name][kernel_hmin:kernel_hmax, kernel_wmin:kernel_wmax] = [255, 0, 0]  # blue
    debugger.save_all_imgs(debug_dir, prefix='{}'.format(opt.img_id))



    opt.prefix = '{}masked'.format(opt.img_id)
    detector.run(inp_image)
    return




if __name__ == '__main__':
    opt = opts().parse()
    opt.not_prefetch_test = True
    opt.prefix = opt.img_id
    error_bound_saliency(opt, opt.img_id)
