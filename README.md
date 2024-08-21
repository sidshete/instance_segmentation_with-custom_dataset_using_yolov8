# instance_segmentation_with-custom_dataset_using_yolov8


YOLOv8 Architecture Overview

Main components of YOLOv8 architecture

    1) Backbone Network - CSPDarknet53
    2) Neck and Head Structures
    3) Detection Head
    
Key features of YOLO model

    1. Optimized Accuracy-Speed Tradeoff
    2. Variety of Pre-trained Models
## Steps to execute the script
Step 1. Preprocessing (inspection_&_preprocess.ipynb)
          
      In the inspection_&_preprocess.ipynb file
        a. We first inspect the data and understand the data provided.
        b. Preprocess the dataset like resizing the images and masks, renaming and cleaning the data
        c. Once the data is preprocessed, we convert the dataset to either COCO or YOLO format.
        d. The data is split into train, test, val with its annotations
        e. To the split data we apply data augmentations techniques. In our case we use albumentations. Note: transformations applied only to train data.
        f. The dataset is finally loaded using pytorch dataloaders, ready to be trained

Step 2. Training (yv8_train.py)

        a. We train the model with pretrained weights. We have four prediction weights: YOLOv8n-seg, se-seg, m-seg, l-seg, and x-seg. The reason for selecting this weight is that n-seg provides lower mAP while faster inference speed, whereas x-seg provides higher mAP but slower speed. There is a tradeoff, thus, to strike a compromise, we employ YOLOv8m-seg.
        b. To train the model on custom dataset requires fine tuning of the model for better accuracy. In our case we utilze AutoML techniques to train the model.  We use hyperopt where we provide a seach space for hyperparameters and we provide a range of hyper params. The model trains on every combiantion of hyperparams and provides the final best model weights with the best hyperparams chosen. This saves us a time to manually tune the prams and train separately.  In addition it uses SGD optimizer for training.
        c. Loss Function: The primary loss function used in YOLOv8 is a combination of localization loss, confidence loss, and class probability loss, and segmentation loss as defined in the YOLO series of papers. This loss function balances the detection accuracy and localization precision necessary for instance segmentation tasks.
    • Metrics
    • Mean Average Precision (mAP): Measures the accuracy of object localization and classification.
    • Precision and Recall: Quantifies the model's ability to detect objects accurately and avoid false positives.
    
Once the training is done, all the model weights are stored int the results folder categorized into the folder names with the hyper params they were trained using hyperopt.
• The training, testing logs along with the predictions are available with the weights in teh respective folders saved in results.
    
Step 3. Evaluation(yv8_eval.ipynb)

    a. Additionally this file lets you load the model and test on any other images and visualize the prediction based of confidence.( eg conf=0.45).
    b. All the models can also be evaluated on validation dataset and check model performances.
    c. The model can then be finally loaded and can be converted to ONNX format to deploy on the edge devices.
    d. One can make the predictions on test images and compare the inference time in the last module of this notebook.


![results](https://github.com/user-attachments/assets/b4574581-74bf-40ea-a892-406f95db68ac)

![val_batch0_labels](https://github.com/user-attachments/assets/57df06b9-dff9-4e87-8d2b-30c2bfa3a017)


