model = dict(
    type='FasterRCNN',       #model type
    pretrained='modelzoo://resnet50',   # pretrained model
    backbone=dict(
        type='ResNet',      #backbone type
        depth=50,			# number of layers the network
        num_stages=4,		# the number of the stage of resnet
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,				# the number of frozen stage, which means this stage does not update the parameters
        style='pytorch'),
    neck=dict(
        type='FPN',			 # neck type
        in_channels=[256, 512, 1024, 2048], # the channels of each stage
        out_channels=256,	# output channel
        num_outs=5),		# output feature layer number
    rpn_head=dict(
        type='RPNHead',		# RPN type
        in_channels=256,	# RPN netwrok input channels
        feat_channels=256,	# channels of feature layer
        anchor_scales=[8],	# anchor's baselen，baselen = sqrt(w*h)，w and h are width and height of anchor
        anchor_ratios=[1.0, 1.5, 2.0,2.5,3.0],	#ratio of anchor
        anchor_strides=[4, 8, 16, 32, 64],		#step size at feature map layer
        target_means=[.0, .0, .0, .0],			# mean
        target_stds=[1.0, 1.0, 1.0, 1.0],		# variance
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',  			# RoIExtractor type
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,			# threshold of positive samples
            neg_iou_thr=0.3, 			# threshold of negative samples
            min_pos_iou=0.3,			# min iou of positive samples, if the max iou is smaller than 0.3
                                        # then ignores all anchors，otherwise keep the anchor with max iou
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',		# positive negative samplers
            num=256,					# number of samplers
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),					# debug mode
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',		# RCNN network positive and negative split
            pos_iou_thr=0.5,			# threshold of positive iou
            neg_iou_thr=0.5,			# iou threshold of negative
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=200)
        								# max_per_img means the number of output det bbox
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'CrowdDataset'		#
data_root = '/export/home/wyin/CenterNet/CenterNet/data/crowd/'		#
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),#(1665,1000)(1456,875),(1248,750),1040,625,(666,400),#(1333, 800),(888,533),(832,500)
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/crowd_train.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/crowd_val.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/crowd_val.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# if #gpu is 8,lr=0.02, if #gpu is 4,lr=0.01
# 2 gpus ==> lr = 0.005

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=8) #save model every 8 epoch
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 80
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/faster_rcnn_r50_fpn_ours'
load_from = None
resume_from = None
workflow = [('train', 1)]


