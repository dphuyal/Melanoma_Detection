import pandas as pd
import numpy as np
import os
import random
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from sklearn import model_selection
import albumentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from config import *

# GPU check
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def clean_train_df(train_df, train_df2):
    train_df = pd.read_csv(os.path.join(TRAIN_CSV_PATH_20,'train.csv'))
    train_df['filepath'] = train_df['image_name'].apply(lambda x: os.path.join(TRAIN_CSV_PATH_20, f'train', f'{x}.jpg'))
    train_df2['filepath'] = train_df2['image_name'].apply(lambda x: os.path.join(TRAIN_CSV_PATH, 'train/train', x+'.jpg'))

    # concatenate info which is not available in 2020 dataset
    common_images = train_df['image_name'].unique()
    print("Common images: ", len(common_images))
    new_data = train_df2[~train_df2['image_name'].isin(common_images)]

    # merge all together
    train_df = pd.concat([train_df, new_data]).reset_index(drop=True)
    print("The length of entire dataset: ", len(train_df))

    # rename the column names on train and test set
    train_df = train_df.rename(columns={"anatom_site_general_challenge": "anatomy"})
    train_df = train_df.rename(columns={"age_approx": "age"})

    cols_drop = ['benign_malignant', 'tfrecord', 'diagnosis', 'width', 'height']
    for drop in cols_drop:
        if drop in train_df.columns:
            train_df.drop([drop], axis=1, inplace=True)

    # impute the missing data on train set
    train_df['sex'].fillna("male", inplace = True) 
    train_df['age'].fillna(0, inplace = True) 
    train_df['anatomy'].fillna("torso", inplace = True)
    train_df['patient_id'].fillna(0, inplace = True) # IP_4382720

    to_encode = ['age', 'sex', 'anatomy']
    encoded_all = []

    label_encoder = LabelEncoder()

    for column in to_encode:
        encoded = label_encoder.fit_transform(train_df[column])
        encoded_all.append(encoded)
    
    train_df['age'] = encoded_all[0]
    train_df['sex'] = encoded_all[1]
    train_df['anatomy'] = encoded_all[2]
    
    norm_clean = preprocessing.normalize(train_df[['age', 'sex', 'anatomy']])

    train_df['age'] = norm_clean[:,0]
    train_df['sex'] = norm_clean[:,1] 
    train_df['anatomy'] = norm_clean[:,2]

    return train_df.reset_index()

def clean_test_df(test_df):
    test_df = pd.read_csv(os.path.join(TEST_CSV_PATH_20,'test.csv'))
    test_df['filepath'] = test_df['image_name'].apply(lambda x: os.path.join(TEST_CSV_PATH_20, f'test', f'{x}.jpg'))

    # rename the column names on test set
    test_df = test_df.rename(columns={"anatom_site_general_challenge": "anatomy"})
    test_df = test_df.rename(columns={"age_approx": "age"})

    # drop unneccessary columns
    if 'width' in test_df.columns : test_df.drop(['width'], axis=1, inplace=True)
    if 'height' in test_df.columns : test_df.drop(['height'], axis=1, inplace=True)

    # impute missing data on test set
    test_df['sex'].fillna("male", inplace = True) 
    test_df['age'].fillna(50, inplace = True) # median age is 50
    test_df['anatomy'].fillna("torso", inplace = True)
    test_df['patient_id'] = test_df['patient_id'].fillna(0)

    to_encode = ['age', 'sex', 'anatomy']
    encoded_all = []

    label_encoder = LabelEncoder()

    for column in to_encode:
        encoded = label_encoder.fit_transform(test_df[column])
        encoded_all.append(encoded)
    
    test_df['age'] = encoded_all[0]
    test_df['sex'] = encoded_all[1]
    test_df['anatomy'] = encoded_all[2]  
    
    norm_clean = preprocessing.normalize(test_df[['age', 'sex', 'anatomy']])

    test_df['age'] = norm_clean[:,0]
    test_df['sex'] = norm_clean[:,1] 
    test_df['anatomy'] = norm_clean[:,2]
    
    return test_df.reset_index()

# custom dataset 
class Melanoma_Dataset(Dataset):

    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.transforms:
            res = self.transforms(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        row_vals = np.array(self.df.iloc[idx][['age', 'sex', 'anatomy']].values,dtype=np.float32)
        
        image = image.transpose(2, 0, 1)
        if 'target' in self.df.columns.values:
            y = self.df.loc[idx,'target']
        else :
            y = 1
        return torch.tensor(image).float(), torch.tensor(row_vals), torch.tensor([y],dtype=torch.float32)
        
    def __len__(self):
        return self.df.shape[0]


class Microscope(A.ImageOnlyTransform):
    def __init__(self, p: float = 0.5, always_apply=False):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        if random.random() < self.p:
            circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8),
                        (img.shape[0]//2, img.shape[1]//2),
                        random.randint(img.shape[0]//2 - 3, img.shape[0]//2 + 15),
                        (0, 0, 0),
                        -1)

            mask = circle - 255
            img = np.multiply(img, mask)

        return img


def get_transforms(image_size):
    transforms_train = albumentations.Compose([
        Microscope(p=0.5),
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightness(limit=0.2, p=0.75),
        albumentations.RandomContrast(limit=0.2, p=0.75),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.7),

        albumentations.CLAHE(clip_limit=4.0, p=0.7),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albumentations.Resize(image_size, image_size),
        albumentations.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    return transforms_train, transforms_val