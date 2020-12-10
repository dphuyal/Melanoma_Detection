import os
import gc
import pandas as pd
import time
import datetime
import numpy as np
import warnings
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torchvision.models import resnet50
from sklearn import model_selection
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from efficientnet_pytorch import EfficientNet
warnings.filterwarnings("ignore")

from pre_processing import clean_dataframe,Melanoma_Dataset,get_transforms
from models import EfficientNet_Models, Resnet_Model
from utils import set_seed
from config import *

set_seed()
# GPU check
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# created a directory, if it does not exist
os.makedirs(model_path_,exist_ok=True)
os.makedirs(logs_path_,exist_ok=True)

# some paths definition
cwd = os.getcwd()
model_path = os.path.join(cwd, model_path_)
logs_path = os.path.join(cwd, logs_path_)


df = pd.read_csv(os.path.join(TRAIN_CSV_PATH_20,'train.csv'))
df2 = pd.read_csv(os.path.join(TRAIN_CSV_PATH,'train_concat.csv')) # roman's dataset with 5k positive samples
df_test = pd.read_csv(os.path.join(TEST_CSV_PATH_20,'test.csv'))

print(df_test.head(5))
exit()


train_df = clean_dataframe(df,df2,df_test)
test_df = clean_dataframe(df,df2,df_test)
#train_df = train_df.sample(1000)


# GroupKFold
group_fold = GroupKFold(n_split)


# train_len=len(train_df)
folds = group_fold.split(X = np.zeros(len(train_df)), 
                         y = train_df['target'],
                         groups = train_df['patient_id'].tolist()
                        )

for fold, (train_index, valid_index) in enumerate(folds):
    print('=' * 20, 'Fold', fold, '=' * 20) 

    # best_val_accuracy = 0 # best validation score in the current fold
    best_ROC = 0
    patience = es_patience # current patience counter    

    df_train = train_df.iloc[train_index].reset_index(drop=True)
    df_valid = train_df.iloc[valid_index].reset_index(drop=True)

    # print(len(df_train), len(df_valid), len(df_test))
    # exit()

    transforms_train, transforms_val = get_transforms(image_size)

    train_dataset = Melanoma_Dataset(df_train,transforms=transforms_train)
    valid_dataset = Melanoma_Dataset(df_valid,transforms=transforms_val)
    test_dataset = ScancerDataset(test_df,transforms=transforms_val)
    
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
        )
    valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8
        )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    if model_type == 'efficientnet':
        model_name = eff_dict[eff_type]
        model=EfficientNet_Models(output_size=output_size, num_cols=num_cols, model_name=model_name)
    elif model_type == 'resnet':
        model = Resnet_Model(output_size=output_size, num_cols=num_cols)
    else:
        print("Wrong model name")
    model.to(device)
    
    '''
    # run two GPUs parallely
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    '''

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max',
    patience=1,
    verbose=True,
    factor=0.2
    )
    criterion = nn.BCEWithLogitsLoss()
    
    # print('Total tl ',len(train_loader))
    
    for epoch in range(epochs):
        start_time = time.time()
        correct = 0
        running_loss = 0 # loss in each epochs
        model.train()
        count=0

        for tl,train_batch in enumerate(train_loader):
            optimizer.zero_grad()
            train_img, train_csv, train_targ = train_batch
            img,csv,y = train_img.float().to(device), train_csv.to(device), train_targ.to(device)
            yhat = model(img, csv)
            loss = criterion(yhat, y.reshape(-1,1).float())
            loss.backward()
            optimizer.step()
            pred = torch.round(torch.sigmoid(yhat))  # round off sigmoid to obtain predictions
            # total_predictions += y.size(0)

            correct  += (y.squeeze().cpu() ==pred.squeeze().cpu()).sum().item()
            running_loss += loss.item()

        # train_accuracy = (correct/total_predictions) * 100
        train_accuracy = correct / len(train_index)
        running_loss /= len(train_loader)
        end_time = time.time()
        # print('Training Loss: ', running_loss, 'Time: ', round(end_time - start_time, 3), 's')
        # print('Training Accuracy: ', round(train_accuracy,3), '%')


        # validating on our validation dataset
        model.eval()

        # matrix to store evaluation predictions
        val_predicts = torch.zeros((len(valid_index), 1), dtype=torch.float32, device=device)

        # disable gradients, no optimization required for evaluation
        with torch.no_grad():
            for k, val_batch in enumerate(valid_loader):
                val_img, val_csv, val_targ = val_batch
                img,csv,y = val_img.to(device), val_csv.to(device), val_targ.to(device)
                
                z_val = model(img, csv)
                val_pred = torch.sigmoid(z_val)
                
                val_predicts[k*valid_loader.batch_size : k*valid_loader.batch_size + img.shape[0]] = val_pred

            # calculate validation accuracy
            val_accuracy = accuracy_score(df_valid['target'].values, torch.round(val_predicts.cpu()))

            # calculate ROC
            val_roc = roc_auc_score(df_valid['target'].values, torch.round(val_predicts.cpu()))
            
            # calculate train and eval time
            duration = str(datetime.timedelta(seconds=time.time() - start_time))[:7]

            # append info to .txt file
            with open(f"new_logs/{model_name}.txt", 'a+') as f:
                print('Fold: {} | Epoch: {}/{} | Training Loss: {:.3f} | Train Acc: {:.3f} | Valid Acc: {:.3f} | ROC: {:.3f} | Training time: {}'.format(
                fold+1,
                epoch+1, 
                epochs, 
                running_loss, 
                train_accuracy, 
                val_accuracy, 
                val_roc, 
                duration), file=f)

            # prints on the console
            print('Epoch: {}/{} | Training Loss: {:.3f} | Train acc: {:.3f} | Val acc: {:.3f} | Val roc_auc: {:.3f} | Training time: {}'.format(
            epoch + 1,
            epochs, 
            running_loss, 
            train_accuracy, 
            val_accuracy, 
            val_roc, 
            duration))

            # update scheduler, updates learning_rate
            scheduler.step(val_roc)

            # update best_ROC
            if not best_ROC: # if best_roc = None
                best_ROC = val_roc
                save_filename = os.path.join(model_path, model_name + "_fold{}_epoch{}_ROC_{:.3f}.pth".format(fold+1,epoch+1,val_roc))
                torch.save(model, save_filename)
                continue

            if val_roc > best_ROC:
                best_ROC = val_roc # reset best_ROC to val_roc 
                patience = es_patience # reset patience
                
                # save_filename = os.path.join(model_path, model_name + "_fold_{}.pth".format(fold)) 
                save_filename = os.path.join(model_path, model_name + "_fold{}_epoch{}_ROC_{:.3f}.pth".format(fold+1,epoch+1,val_roc)) 
                # save model with highest ROC
                torch.save(model, save_filename)
            else: 
                patience -= 1
                if patience == 0:
                    # write to the file
                    with open(f"new_logs/lr0.0005_Dout0.3{model_name}.txt", 'a+') as f:
                        print('Early stopping no improvement since 3 epochs | Best ROC {:.3f}'.format(best_ROC), file=f)
                    # print on the console
                    print('Early stopping no improvement since 3 epochs | Best ROC {:.3f}'.format(best_ROC))
                    break
            
    # to prevent memory leaks
    del model, train_dataset, valid_dataset, train_loader, valid_loader
    # garbage collector
    gc.collect()
