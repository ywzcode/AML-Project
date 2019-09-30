from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import torch
import json
import os
import time
import datetime
import torch.utils.data as data


class Crowd(data.Dataset):
    num_classes = 1
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(Crowd, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'crowd')
        self.img_dir = os.path.join(self.data_dir, 'images')
        _ann_name = {'train': 'train', 'val': 'val'}
        self.annot_path = os.path.join(
            self.data_dir, 'annotations',
            'crowd_{}.json').format(_ann_name[split])
        self.max_objs = 200 # max person per image is 400 (<2% image contains person>200)
        self.class_name = ['__background__', "person"]
        self._valid_ids = np.arange(1, 2, dtype=np.int32)
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.split = split
        self.opt = opt

        print('==> initializing pascal {} data.'.format(_ann_name[split]))
        self.coco = coco.COCO(self.annot_path)
        self.images = sorted(self.coco.getImgIds())
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir,pre_fix=''):
        json.dump(self.convert_eval_format(results),
                  open('{}/{}results.json'.format(save_dir,pre_fix), 'w'))

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        self.eval('{}/results.json'.format(save_dir))
    def eval(self,coco_file):
        coco_dets = self.coco.loadRes(coco_file)
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.params.maxDets = [200]
        coco_eval.params.iouThrs = np.array([0.5])
        coco_eval.params.fppiThrs = np.logspace(-2, 0, 9)
        coco_eval.evaluate()
        self.accumulate(coco_eval)
        self.summarize(coco_eval)
        return coco_eval

    def summarize(self,coco_eval):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = coco_eval.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            if ap==1:
                titleStr = 'Average Precision'
                typeStr = '(AP)'
            elif ap==0:
                titleStr = 'Average Recall'
                typeStr = '(AR)'
            elif ap==2:
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
            elif ap==0:
                # dimension of recall: [TxKxAxM]
                s = coco_eval.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            elif ap==2: # mr
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
            stats_all = np.zeros((len(coco_eval.params.maxDets),12))
            for i_maxDet,maxDet in enumerate(coco_eval.params.maxDets):
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


    def accumulate(self,coco_eval, p=None):
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
        miss_rate = -np.ones((T,F, K, A, M))
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
                        pr = pr.tolist();q = q.tolist()

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
                        mr = mr.tolist();f = f.tolist()
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
            'miss_rate':miss_rate,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))
