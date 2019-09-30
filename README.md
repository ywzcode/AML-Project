# AML Project: Object Detection In Crowded Scenes

In this project, we modify 3 representative state-of-art detectors and train them on CrowdHuman dataset. Faster R-CNN with FPN, RepPoints and Object as Points are investigated. They represent three typical new ideas in general object detection problem respectively: classical anchor-based two-stage detector with feature pyramid, deformable-based anchor free two-stage detector and anchor free one-stage detector

## Installation

### For Center Net

After install Anaconda:

1. [Optional but recommended] create a new conda environment. 

    ```
    conda create --name CenterNet python=3.6
    ```

    And activate the environment.

    ```
    conda activate CenterNet
    ```

2. Install pytorch0.4.1:

    ```
    conda install pytorch torchvision -c pytorch
    ```

3. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ```
    # COCOAPI=/path/to/clone/cocoapi
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
    cd $COCOAPI/PythonAPI
    make
    python setup.py install --user
    ```

4. Clone this repo:

    ```
    CenterNet_ROOT=/path/to/clone/CenterNet
    git clone https://github.com/xingyizhou/CenterNet $CenterNet_ROOT
    ```

5. Install the requirements

    ```
    pip install -r requirements.txt
    ```

6. Compile and install deformable convolutional (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/master)).


### For mmdetection

```
python setup.py develop
```

## Dataset preparation

### Crowd

- Use the `./src/tools/crowd/get_crowd.sh` to download the crowd dataset and 

- Place the data (or create symlinks) to make the data folder like:

    ```
    ${CenterNet_ROOT}
    |-- data
    `-- |-- crowd
        `-- |-- annotations
            |   |-- crowd_val.json
            |   |-- crowd_train.json
            `-- images
                |-- 273271,1a0d6000b9e1f5b7.jpg
                |-- ...
    ```

## Training

### training Center Net
Use the `./experiments/train_crowd.sh`for training and testing and saliency analysis.

Analysis and download the dataset, use the file under the `src\tools\crowd`.


## Reference

Center Net: https://github.com/xingyizhou/CenterNet

MMdetection: https://github.com/open-mmlab/mmdetection