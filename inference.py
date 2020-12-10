import os
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from pre_processing import clean_test_df,Melanoma_Dataset,get_transforms
from models import EfficientNet_Models, Resnet_Model
from utils import set_seed
from config import *

set_seed()

df_test = pd.read_csv(os.path.join(TEST_CSV_PATH_20,'test.csv'))

test_df = clean_test_df(df_test)
print(test_df.head(5))

# transformations for images
transforms_val = get_transforms(image_size)
test_dataset = Melanoma_Dataset(test_df,transforms=transforms_val)

# dataloader
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=8
)
print(len(test_loader))

# load the model with best ROC in a fold
best_model = torch.load('saved_models_efficientnet-b2/efficientnet-b2_fold3_epoch3_ROC_0.947.pth')

# model eval
best_model.eval()
# average test run
AVE_TEST = 3
# predictions matrix
predict_sub = torch.zeros(size=len(test_df),1), dtype=torch.float32, device=device)

with torch.no_grad():
    for i in range(AVE_TEST):
        for k, test_batch in enumerate(test_loader):
            test_img, test_csv = test_batch
            img, csv = test_img.to(device), test_csv.to(device)
            yhat = best_model(img, csv)
            # covert to probablities
            yhat = torch.sigmoid(out)

            # adds the prediction to the matrix created above
            predict_sub[k*img.shape[0] : k*img.shape[0] + img.shape[0]] += yhat


    # divide predictions by AVE_TEST to get the average test accuracy
    predict_sub /= AVE_TEST
        
predict_sub = predict_sub.cpu().numpy().reshape(-1,)

# import submission file
sample_sub = pd.read_csv(os.path.join(SUB_FILE_PATH_20,'sample_submission.csv'))

sample_sub['target'] = predict_sub
# 'effnetb2_500+250_64_test.csv' is submitted on kaggle for private and public LB score
sample_sub.to_csv(f'effnetb2_500+250_64_test.csv', index=False)