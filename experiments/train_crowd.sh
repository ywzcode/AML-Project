cd src

## for faster r-cnn 
# train
python mm_main.py ../experiments/faster_rcnn_r50_fpn.py
# test
python tools/test.py ../experiments/faster_rcnn_r50_fpn.py ./work_dirs/faster_rcnn_r50_fpn/latest.pth --json_out ./results/faster_1333_800.json


## for reppoints
# train
python mm_main.py ../experiments/reppoints_moment_r50_fpn_2x.py
# test
python tools/test.py ../experiments/reppoints_moment_r50_fpn_2x.py ./work_dirs/reppoints_moment_r50_fpn_2x/latest.pth --json_out ./results/reppoints_1333_800.json


## for Center Net
# train
python main.py ctdet  --dataset crowd --exp_id crowd_dla_512_b64 --batch_size 64 --lr 2.5e-4 --gpus 1,2,3 --num_workers 16
# test
python test.py ctdet --dataset crowd --exp_id crowd_dla_512_b64 --input_res 1024 --resume



