from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time

#
from datasets.dataset_factory import dataset_factory


from opts import opts
from pycocotools.cocoeval import COCOeval
import copy
import matplotlib.pyplot as plt
import pycocotools.coco as coco
import datetime
from pycocotools import mask as maskUtils
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (8.0, 8.0)
# def bbox_iou(box1, box2,input_type='xywh'):
#     "box: [xmin,ymin,xmax,ymax]"
#     xywh_to_xyxy = lambda box: [box[0] - box[2] / 2 + 0.5, box[1] - box[3] / 2 + 0.5, box[0] + box[2] / 2 - 0.5, box[1] + box[3] / 2 - 0.5]
#     box1 = xywh_to_xyxy(box1)
#     box2 = xywh_to_xyxy(box2)
#     area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
#     area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
#     inter = max(min(box1[2], box2[2]) - max(box1[0], box2[0]) + 1, 0) * \
#             max(min(box1[3], box2[3]) - max(box1[1], box2[1]) + 1, 0)
#     iou = 1.0 * inter / (area1 + area2 - inter)
#     return iou


def summarize(coco_eval):
    '''
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    '''

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = coco_eval.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        if ap == 1:
            titleStr = 'Average Precision'
            typeStr = '(AP)'
        elif ap == 0:
            titleStr = 'Average Recall'
            typeStr = '(AR)'
        elif ap == 2:
            titleStr = 'Average Miss Rate'
            typeStr = 'MR(-2)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = coco_eval.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        elif ap == 0:
            # dimension of recall: [TxKxAxM]
            s = coco_eval.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        elif ap == 2:  # mr
            s = coco_eval.eval['miss_rate']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        return mean_s

    def _summarizeDets():
        stats_all = np.zeros((len(coco_eval.params.maxDets), 12))
        for i_maxDet, maxDet in enumerate(coco_eval.params.maxDets):
            stats = np.zeros((12,))

            stats[0] = _summarize(1, iouThr=.5, maxDets=coco_eval.params.maxDets[i_maxDet])
            stats[1] = _summarize(1, iouThr=.5, areaRng='small', maxDets=coco_eval.params.maxDets[i_maxDet])
            stats[2] = _summarize(1, iouThr=.5, areaRng='medium', maxDets=coco_eval.params.maxDets[i_maxDet])
            stats[3] = _summarize(1, iouThr=.5, areaRng='large', maxDets=coco_eval.params.maxDets[i_maxDet])

            stats[4] = _summarize(0, iouThr=.5, maxDets=coco_eval.params.maxDets[i_maxDet])
            stats[5] = _summarize(0, iouThr=.5, areaRng='small', maxDets=coco_eval.params.maxDets[i_maxDet])
            stats[6] = _summarize(0, iouThr=.5, areaRng='medium', maxDets=coco_eval.params.maxDets[i_maxDet])
            stats[7] = _summarize(0, iouThr=.5, areaRng='large', maxDets=coco_eval.params.maxDets[i_maxDet])

            stats[8] = _summarize(2, iouThr=.5, maxDets=coco_eval.params.maxDets[i_maxDet])
            stats[9] = _summarize(2, iouThr=.5, areaRng='small', maxDets=coco_eval.params.maxDets[i_maxDet])
            stats[10] = _summarize(2, iouThr=.5, areaRng='medium', maxDets=coco_eval.params.maxDets[i_maxDet])
            stats[11] = _summarize(2, iouThr=.5, areaRng='large', maxDets=coco_eval.params.maxDets[i_maxDet])
            stats_all[i_maxDet] = stats
        return stats_all

    if not coco_eval.eval:
        raise Exception('Please run accumulate() first')
    iouType = coco_eval.params.iouType
    summarize = _summarizeDets
    coco_eval.stats = summarize()


def accumulate(coco_eval, p=None):
    '''
    Accumulate per image evaluation results and store the result in self.eval
    :param p: input params for evaluation
    :return: None
    '''
    print('Accumulating evaluation results...')
    tic = time.time()
    if not coco_eval.evalImgs:
        print('Please run evaluate() first')
    # allows input customized parameters
    if p is None:
        p = coco_eval.params
    p.catIds = p.catIds if p.useCats == 1 else [-1]
    T = len(p.iouThrs)
    R = len(p.recThrs)
    K = len(p.catIds) if p.useCats else 1
    A = len(p.areaRng)
    M = len(p.maxDets)
    precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
    recall = -np.ones((T, K, A, M))

    F = len(p.fppiThrs)
    miss_rate = -np.ones((T, F, K, A, M))
    scores = -np.ones((T, R, K, A, M))

    # create dictionary for future indexing
    _pe = coco_eval._paramsEval
    catIds = _pe.catIds if _pe.useCats else [-1]
    setK = set(catIds)
    setA = set(map(tuple, _pe.areaRng))
    setM = set(_pe.maxDets)
    setI = set(_pe.imgIds)
    # get inds to evaluate
    k_list = [n for n, k in enumerate(p.catIds) if k in setK]
    m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
    a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
    i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
    I0 = len(_pe.imgIds)
    A0 = len(_pe.areaRng)
    # retrieve E at each category, area range, and max number of detections
    for k, k0 in enumerate(k_list):
        Nk = k0 * A0 * I0
        for a, a0 in enumerate(a_list):
            Na = a0 * I0
            for m, maxDet in enumerate(m_list):
                E = [coco_eval.evalImgs[Nk + Na + i] for i in i_list]
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue
                dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.
                inds = np.argsort(-dtScores, kind='mergesort')
                dtScoresSorted = dtScores[inds]

                dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                gtIg = np.concatenate([e['gtIgnore'] for e in E])
                npig = np.count_nonzero(gtIg == 0)
                if npig == 0:
                    continue
                tps = np.logical_and(dtm, np.logical_not(dtIg))
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    nd = len(tp)
                    rc = tp / npig
                    pr = tp / (fp + tp + np.spacing(1))
                    mr = 1. - rc
                    ffpi = fp / len(coco_eval.cocoGt.imgs)
                    q = np.zeros((R,))
                    ss = np.zeros((R,))

                    if nd:
                        recall[t, k, a, m] = rc[-1]
                    else:
                        recall[t, k, a, m] = 0

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    pr = pr.tolist();
                    q = q.tolist()

                    for i in range(nd - 1, 0, -1):
                        if pr[i] > pr[i - 1]:
                            pr[i - 1] = pr[i]

                    inds = np.searchsorted(rc, p.recThrs, side='left')
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = pr[pi]
                            ss[ri] = dtScoresSorted[pi]
                    except:
                        pass
                    precision[t, :, k, a, m] = np.array(q)
                    scores[t, :, k, a, m] = np.array(ss)

                    f = np.zeros((F,))
                    mr = mr.tolist();
                    f = f.tolist()
                    for i in range(nd - 1, 0, -1):
                        if mr[i] > mr[i - 1]:
                            mr[i - 1] = mr[i]
                    inds = np.searchsorted(ffpi, p.fppiThrs, side='left')
                    try:
                        for ri, pi in enumerate(inds):
                            f[ri] = mr[pi]
                    except:
                        pass
                    miss_rate[t, :, k, a, m] = np.array(f)
    coco_eval.eval = {
        'params': p,
        'counts': [T, R, K, A, M],
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'precision': precision,
        'recall': recall,
        'scores': scores,
        'miss_rate': miss_rate,
    }
    toc = time.time()
    print('DONE (t={:0.2f}s).'.format(toc - tic))


def adjustImg(cocoeval, imgId, catId, aRng, maxDet, up_type):
    '''
    perform evaluation for single category and image
    :return: dict (single image results)
    '''
    p = cocoeval.params
    if p.useCats:
        gt = cocoeval._gts[imgId, catId]
        dt = cocoeval._dts[imgId, catId]
    else:
        gt = [_ for cId in p.catIds for _ in cocoeval._gts[imgId, cId]]
        dt = [_ for cId in p.catIds for _ in cocoeval._dts[imgId, cId]]
    if len(gt) == 0 and len(dt) == 0:
        return None

    for g in gt:
        if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
            g['_ignore'] = 1
        else:
            g['_ignore'] = 0

    # sort dt highest score first, sort gt ignore last
    gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
    gt = [gt[i] for i in gtind]
    dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
    dt = [dt[i] for i in dtind[0:maxDet]]
    iscrowd = [int(o['iscrowd']) for o in gt]
    # load computed ious
    ious = cocoeval.ious[imgId, catId][:, gtind] if len(cocoeval.ious[imgId, catId]) > 0 else cocoeval.ious[
        imgId, catId]

    T = len(p.iouThrs)
    G = len(gt)
    D = len(dt)
    gtm = np.zeros((T, G))
    dtm = np.zeros((T, D))
    gtIg = np.array([g['_ignore'] for g in gt])
    dtIg = np.zeros((T, D))
    if not len(ious) == 0:
        for tind, t in enumerate(p.iouThrs):
            for dind, d in enumerate(dt):
                # information about best match so far (m=-1 -> unmatched)
                iou = min([t, 1 - 1e-10])
                m = -1
                for gind, g in enumerate(gt):
                    # if this gt already matched, and not a crowd, continue
                    if gtm[tind, gind] > 0 and not iscrowd[gind]:
                        continue
                    # if dt matched to reg gt, and on ignore gt, stop
                    if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                        break
                    # continue to next gt unless better match made
                    if ious[dind, gind] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = ious[dind, gind]
                    m = gind
                # if match made store id of match for both dt and gt
                if m == -1:
                    continue
                dtIg[tind, dind] = gtIg[m]
                dtm[tind, dind] = gt[m]['id']
                gtm[tind, m] = d['id']
    # set unmatched detections outside of area range to ignore
    a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]).reshape((1, len(dt)))
    dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
    # store results for given image and category
    if up_type == 'perfect nms':
        assigned_bboxes = [cocoeval.cocoDt.anns[dtId]['bbox'] for dtId in gtm[0] if dtId>0]
        for dtId in [d['id'] for d in dt]:
            if not dtId in gtm:
                un_assigned_bbox = [cocoeval.cocoDt.anns[dtId]['bbox'],]
                iou_with_assigned_det = np.max(maskUtils.iou(un_assigned_bbox,assigned_bboxes,[0]*len(assigned_bboxes)))
                if iou_with_assigned_det >0:
                    cocoeval.cocoDt.anns[dtId]['score'] = 0.0
    elif up_type == 'better nms':
        dtIds = [d['id'] for d in dt]
        gtIds = [g['id'] for g in gt]
        assigned_bboxes_ids = [dtId for dtId in gtm[0] if dtId>0]
        assigned_bboxes = [cocoeval.cocoDt.anns[dtId]['bbox'] for dtId in assigned_bboxes_ids]
        for dtId in [d['id'] for d in dt]:
            if not dtId in gtm:
                un_assigned_bbox = [cocoeval.cocoDt.anns[dtId]['bbox'],]
                iou_with_assigned_bboxes = maskUtils.iou(un_assigned_bbox,assigned_bboxes,[0]*len(assigned_bboxes))
                if np.max(iou_with_assigned_bboxes)<0.01:
                    continue
                assigned_bbox_id = assigned_bboxes_ids[np.argmax(iou_with_assigned_bboxes)]
                assigned_bbox = [cocoeval.cocoDt.anns[assigned_bbox_id]['bbox'],]
                assigned_gt_id = dtm[0][dtIds.index(assigned_bbox_id)]
                assigned_gt_box = [cocoeval.cocoGt.anns[assigned_gt_id]['bbox'],]
                corresponding_assigned_bbox_iou_with_gt = ious[dtIds.index(assigned_bbox_id),gtIds.index(assigned_gt_id)]

                iou_with_assigned_gt = maskUtils.iou(un_assigned_bbox,assigned_gt_box,[0,])
                if iou_with_assigned_gt >0.5*corresponding_assigned_bbox_iou_with_gt:
                    cocoeval.cocoDt.anns[dtId]['score'] = 0.0
    elif up_type =='perfect center':
        for dtId in [d['id'] for d in dt]:
            if not dtId in gtm:
                cocoeval.cocoDt.anns[dtId]['score'] = 0.0
    elif up_type =='better center':
        for dtId in [d['id'] for d in dt]:
            if not dtId in gtm:
                cocoeval.cocoDt.anns[dtId]['score'] = cocoeval.cocoDt.anns[dtId]['score']*0.5
    elif up_type == 'better regress' or up_type == 'perfect regress':
        dtIds = [d['id'] for d in dt]
        for dtId in [d['id'] for d in dt]:
            if dtId in gtm:
                gtId = dtm[0][dtIds.index(dtId)]
                cocoeval.cocoDt.anns[dtId]['bbox'] = cocoeval.cocoGt.anns[gtId]['bbox']
    elif up_type == 'scoring':
        for dind, d in enumerate(dt):
            iou = max([ious[dind, gind] for gind, g in enumerate(gt)])
            cocoeval.cocoDt.anns[d['id']]['score'] = iou
    return {
        'image_id': imgId,
        'category_id': catId,
        'aRng': aRng,
        'maxDet': maxDet,
        'dtIds': [d['id'] for d in dt],
        'gtIds': [g['id'] for g in gt],
        'dtMatches': dtm,
        'gtMatches': gtm,
        'dtScores': [d['score'] for d in dt],
        'gtIgnore': gtIg,
        'dtIgnore': dtIg,
    }


def adjust(cocoeval, up_type):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    tic = time.time()
    print('Running per image adjust...')
    p = cocoeval.params
    # add backward compatibility if useSegm is specified in params
    if not p.useSegm is None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    print('Adjust annotation type *{}*'.format(up_type))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    cocoeval.params = p

    cocoeval._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = cocoeval.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = cocoeval.computeOks
    cocoeval.ious = {(imgId, catId): computeIoU(imgId, catId) \
                     for imgId in p.imgIds
                     for catId in catIds}

    maxDet = p.maxDets[-1]
    cocoeval.evalImgs = [adjustImg(cocoeval, imgId, catId, areaRng, maxDet, up_type)
                         for catId in catIds
                         for areaRng in [p.areaRng[0], ]
                         for imgId in p.imgIds
                         ]
    cocoeval._paramsEval = copy.deepcopy(cocoeval.params)
    toc = time.time()
    print('DONE (t={:0.2f}s).'.format(toc - tic))


def dump_coco(coco, save_name):
    detections = []
    for i, result in coco.anns.items():
        detection = {
            "image_id": result['image_id'],
            "category_id": result['category_id'],
            "bbox": result['bbox'],
            "score": float("{:.2f}".format(result['score']))
        }
        if result['score'] > 1e-10:
            detections.append(detection)
    json.dump(detections,
              open(save_name, 'w'))
    return


def upperbound_analysis(gt_file, result_json_file, up_type='nms'):
    """

    :param result_json_file:
    :param up_type: "nms" , "regression","scoring"
    '''
    IOU(det,gts) = max( all IOU(det,gt) )
    perfect center: set all unassigned detection's score to 0. remove all false positive
    better center :  set all unassigned detection's score to 0.5x.
    scoring: set all detection's score = IOU(det,gts)
    perfect nms : for unassigned detection, if it's IOU with assigned detection result>0.5 * coresponding det, IOU(det,gt), set it's score to 0.
    better nms: for unassigned detection, if it's IOU with assigned detection result>0, set it's score to 0.
    perfect regress: set all detection IOU(det,gts) > 0.01 to gt box. More precise location regression
    better regress: set all detection IOU(det,gts) > 0.25 to gt box. More precise location regression
    '''
    :return:
    """
    assert up_type in ['perfect center','better center','scoring','perfect nms','better nms','perfect regress','better regress']
    results_file_path = os.path.dirname(result_json_file)
    coco_gt = coco.COCO(gt_file)
    coco_dets = coco_gt.loadRes(('{}/results.json').format(results_file_path))
    coco_eval = COCOeval(coco_gt, coco_dets, "bbox")
    coco_eval.params.maxDets = [200]
    coco_eval.params.iouThrs = np.array([0.5])
    coco_eval.params.fppiThrs = np.logspace(-2, 0, 9)
    if up_type == 'perfect regress':
        coco_eval.params.iouThrs = np.array([0.01])
    elif up_type == 'better regress':
        coco_eval.params.iouThrs = np.array([0.25])

    adjust(coco_eval, up_type)
    save_name = '{}/{}_results.json'.format(results_file_path, up_type)
    dump_coco(coco_eval.cocoDt, save_name)
    return


def eval(gt_file, det_file):
    coco_gt = coco.COCO(gt_file)
    coco_dets = coco_gt.loadRes(det_file)
    coco_eval = COCOeval(coco_gt, coco_dets, "bbox")
    coco_eval.params.maxDets = [200]
    coco_eval.params.iouThrs = np.array([0.5])
    coco_eval.params.fppiThrs = np.logspace(-2, 0, 9)
    coco_eval.evaluate()
    accumulate(coco_eval)
    summarize(coco_eval)
    return coco_eval


if __name__ == '__main__':
    opt = opts().parse()
    # # get det raw result before threshing and cut count  ---- has little effect to final result, so simply use det result


    ### It's unnecessary to build dataset, I only do it to get value of gt_file
    Dataset = dataset_factory[opt.dataset]
    split = 'val' if not opt.trainval else 'test'
    dataset = Dataset(opt, split)
    ## transfer the raw result to different upper-bound result
    result_file = os.path.join(opt.save_dir, 'results.json')
    gt_file = dataset.annot_path
    # up_type_set = ('perfect regress','better regress')
    up_type_set = ('perfect center','scoring','perfect nms','better regress')
    # up_type_set = ('perfect center','better center','scoring','perfect nms','better nms','perfect regress','better regress')
    coco_eval_dic = {}

    # adjust and output new json file
    for up_type in up_type_set:
        upperbound_analysis(gt_file, result_file, up_type=up_type)

    # evaluate
    for up_type in up_type_set:
        print('Evaluate adjusted  *{}*'.format(up_type))
        save_name = '{}/{}_results.json'.format(opt.save_dir, up_type)
        coco_eval_dic[up_type] = eval(gt_file, save_name)
    coco_eval_dic['raw'] = eval(gt_file, result_file)

    all_det = up_type_set + ('raw',)
    # draw PR curve
    plt.figure()
    for up_type in all_det:
        coco_eval = coco_eval_dic[up_type]
        pr = coco_eval.eval['precision'][0, :, 0, 0, -1]
        x = coco_eval.params.recThrs
        plt.plot(x, pr, label='{} AP={:.3f}'.format(up_type, coco_eval.stats[-1][0]))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('CenterNet:Comparsion of different bottleneck on PR curve')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend(loc='lower left', prop={'size': 8})
    plt.savefig('/export/home/wyin/RP.png')

    ## FPPI curve
    plt.figure()
    for up_type in all_det:
        coco_eval = coco_eval_dic[up_type]
        mr = coco_eval.eval['miss_rate'][0, :, 0, 0, -1]
        x = coco_eval.params.fppiThrs
        plt.plot(x, mr, label='{} mMR={:.3f}'.format(up_type, coco_eval.stats[-1][8]))
    plt.xlabel('FFPI')
    plt.ylabel('Miss Rate')
    plt.title('CenterNet:Comparsion of different bottleneck on FFPI curve')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend(loc='upper right', prop={'size': 8})
    plt.savefig('/export/home/wyin/FFPI.png')
