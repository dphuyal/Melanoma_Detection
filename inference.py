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

transforms_val = get_transforms(image_size)
test_dataset = Melanoma_Dataset(test_df,transforms=transforms_val)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=8
)
print(len(test_loader))
exit()