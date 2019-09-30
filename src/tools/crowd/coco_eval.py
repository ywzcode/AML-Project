import matplotlib.pyplot as plt
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import time
import datetime
def summarize(coco_eval):
    '''
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    '''

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = coco_eval.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        if ap==1:
            titleStr = 'Average Precision'
        elif ap==0:
            titleStr = 'Average Recall'
        elif ap==2:
            titleStr = 'Average Miss Rate'
        typeStr = '(AP)' if ap == 1 else '(AR)'
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
        stats = np.zeros((12,))

        stats[0] = _summarize(1, iouThr=.5, maxDets=coco_eval.params.maxDets[-1])
        stats[1] = _summarize(1, iouThr=.5, areaRng='small', maxDets=coco_eval.params.maxDets[-1])
        stats[2] = _summarize(1, iouThr=.5, areaRng='medium', maxDets=coco_eval.params.maxDets[-1])
        stats[3] = _summarize(1, iouThr=.5, areaRng='large', maxDets=coco_eval.params.maxDets[-1])

        stats[4] = _summarize(0, iouThr=.5, maxDets=coco_eval.params.maxDets[-1])
        stats[5] = _summarize(0, iouThr=.5, areaRng='small', maxDets=coco_eval.params.maxDets[-1])
        stats[6] = _summarize(0, iouThr=.5, areaRng='medium', maxDets=coco_eval.params.maxDets[-1])
        stats[7] = _summarize(0, iouThr=.5, areaRng='large', maxDets=coco_eval.params.maxDets[-1])

        stats[8] = _summarize(2, iouThr=.5, maxDets=coco_eval.params.maxDets[-1])
        stats[9] = _summarize(2, iouThr=.5, areaRng='small', maxDets=coco_eval.params.maxDets[-1])
        stats[10] = _summarize(2, iouThr=.5, areaRng='medium', maxDets=coco_eval.params.maxDets[-1])
        stats[11] = _summarize(2, iouThr=.5, areaRng='large', maxDets=coco_eval.params.maxDets[-1])
        return stats

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


annot_path = './crowd_val.json'
det_coco = coco.COCO(annot_path)
coco_dets = det_coco.loadRes('./mmdetection/results/reppoints_1873_1125.bbox.json')
coco_eval = COCOeval(det_coco, coco_dets, "bbox")
coco_eval.params.maxDets = [200]
coco_eval.params.iouThrs = np.array([0.5])
coco_eval.params.fppiThrs = np.logspace(-2,0,9)
coco_eval.evaluate()
accumulate(coco_eval)
summarize(coco_eval)

pr = coco_eval.eval['precision'][0,:,0,0,-1]
x = coco_eval.params.recThrs
plt.plot(x,pr,label='AP={:.3f}'.format(coco_eval.stats[0]))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim(0,1.0)
plt.ylim(0,1.01)
plt.grid(True)
plt.legend(loc='upper right')
plt.show()


#%%

mr = coco_eval.eval['miss_rate'][0, :, 0, 0, -1]
x = coco_eval.params.fppiThrs
plt.loglog(x, mr, label='mMR={:.3f}'.format(coco_eval.stats[8]))
plt.xlabel('FFPI')
plt.ylabel('Miss Rate')
plt.xlim(0, 1.0)
plt.ylim(0, 1.01)
plt.grid(True)
plt.legend(loc='upper right')
plt.show()