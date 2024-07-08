#!/usr/bin/env python3
import os
import shutil
from ultralytics import YOLO
from hyperopt import hp, tpe, Trials, fmin, space_eval

# Set environment variables
os.environ['NUMBAPRO_LIBDEVICE'] = "/usr/local/cuda-11.8/nvvm/libdevice"
os.environ['NUMBAPRO_NVVM'] = "/usr/local/cuda-11.8/nvvm/lib64/libnvvm.so"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize YOLO model for instance segmentation
model = YOLO("yolov8m-seg.pt") 

# Set project and data paths
project_dir = os.getcwd()
data_path = '/path/sshete/sereact_instance/yolov8_instance/yolo_dataset/data.yaml'

# Create results directory 
results_dir = os.path.join(project_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# Define the search space for hyperparameters
space = {
    'lr0': hp.uniform('lr0', 1e-5, 1e-3),
    'lrf': hp.uniform('lrf', 1e-5, 1e-3),
    'batch': hp.choice('batch', [2, 4, 8, 16, 32, 64]),
    'epochs': hp.choice('epochs', [1]),
    'wd': hp.uniform('wd', 0, 5e-4),
}

# Define the objective function for hyperparameter optimization
def objective(params):
    try:
        lr0 = params['lr0']
        lrf = params['lrf']
        batch = params['batch']
        epochs = params['epochs']
        wd = params['wd']
        
        experiment_name = f"model_lr0_{lr0:.1e}_lrf_{lrf:.1e}_b{batch}_epochs{epochs}_wd{wd:.1e}"
        experiment_dir = os.path.join(results_dir, experiment_name)
        
   
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Print the combination of hyperparameters
        print(f"Training with hyperparameters: {params}")
        #path_to_config = '/path/default.yaml'
        # Use (cfg=path_to_config,) in the model.train() arguments below if you want to train on different params apart from default values.
        # Train the model
        model.train(data=data_path, 
                    lr0=lr0, 
                    lrf=lrf, 
                    batch=batch, 
                    epochs=epochs, 
                    weight_decay=wd,
                    project=experiment_dir,
                    name=experiment_name,
                    patience=10)
        
        metrics = model.val()
        
        # Extract validation losses
        box_loss = metrics.box_loss
        cls_loss = metrics.cls_loss
        seg_loss = metrics.seg_loss  
        total_loss = box_loss + cls_loss + seg_loss
        

        for item in os.listdir(experiment_dir):
            s = os.path.join(experiment_dir, item)
            d = os.path.join(experiment_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)


        return {
            'loss': float(total_loss),
            'status': 'ok'
        }

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return {
            'loss': float('inf'),
            'status': 'fail'
        }
    

# Run hyperparameter optimization
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=10, trials=trials)

# Print best hyperparameters and retrain the model
print("Best hyperparameters:", best)
best_params = space_eval(space, best)
print("Retraining model with best hyperparameters...")
print(best_params)

# Define experiment name and directory for the best model
best_experiment_name = f"best_model_lr0_{best_params['lr0']:.1e}_lrf_{best_params['lrf']:.1e}_b{best_params['batch']}_epochs{best_params['epochs']}_wd{best_params['wd']:.1e}"
best_experiment_dir = os.path.join(results_dir, best_experiment_name)

os.makedirs(best_experiment_dir, exist_ok=True)

# Train the model with the best hyperparameters
model.train(data=data_path, 
            lr0=best_params['lr0'], 
            lrf=best_params['lrf'], 
            batch=best_params['batch'], 
            epochs=best_params['epochs'], 
            weight_decay=best_params['wd'],
            project=best_experiment_dir,
            name=best_experiment_name,
            patience=10)

for item in os.listdir(best_experiment_dir):
    s = os.path.join(best_experiment_dir, item)
    d = os.path.join(best_experiment_dir, item)
    if os.path.isdir(s):
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)

print(f"Best model saved in {best_experiment_dir}")
