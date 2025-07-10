import os
import torch
from torch.utils.data import Dataset
import pandas as pd

class WSIFeatDataset(Dataset):
    """Dataset class for classification"""
    def __init__(self,
                 data_csv: pd.DataFrame,
                 feature_dir: str,
                 id_column: str = "Case ID",
                 label_column: str = "consensus_class_simpl",
                 label_names: list = ["LumAll", "Ba/Sq", "Stroma-rich"]
                ):
        super().__init__()
        self.data_csv = data_csv
        self.feature_dir = feature_dir
        self.id_column = id_column
        self.label_column = label_column
        self.label_names = label_names
        self.label_dict = {label_names[i]: i for i in range(len(label_names))}
        self.num_classes = len(label_names)
             
    def __len__(self):
        return len(self.data_csv)
        
    def __getitem__(self, idx):
        sample = self.data_csv.iloc[idx]
        file_name = str(sample[self.id_column])
        features = torch.load(os.path.join(self.feature_dir, file_name + '.pt'), map_location='cpu')
        self.label_dict = {self.label_names[i]: i for i in range(len(self.label_names))}
        label = torch.tensor(self.label_dict[sample[self.label_column]])
        
        return file_name, features, label


class WSIFeatDataset_Reg(Dataset):
    """ Modified Dataset class for multi-point regression"""
    
    def __init__(self,
                 data_csv: pd.DataFrame,
                 feature_dir: str,
                 id_column: str = "Case ID",
                 target_column: str = "combined",
                 class_column: str = None,
                 phase: str = "train", # can be "train" or "val"
                 label_names: list = ["LumAll", "Ba/Sq", "Stroma-rich"]
                ):
        super().__init__()
        self.data_csv = data_csv
        self.feature_dir = feature_dir
        self.id_column = id_column
        self.target_column = target_column  
        self.class_column = class_column
        self.phase = phase  # 0 or 1 for "train" or "val"
        label_names: list = ["LumAll", "Ba/Sq", "Stroma-rich"]
        self.label_names = label_names

        if self.class_column and self.phase == "val":
            self.label_dict = {label_names[i]: i for i in range(len(label_names))}
        else:
            self.label_dict = None
        
        print(f" Phase: {self.phase} | Categorical class mapping: {self.label_dict}")

    def __len__(self):
        return len(self.data_csv)
    
    def __getitem__(self, idx):
        sample = self.data_csv.iloc[idx]
        file_name = str(sample[self.id_column])
        features = torch.load(os.path.join(self.feature_dir, file_name + '.pt'), map_location='cpu')


        if self.phase == "train":
            # Train Phase: regression target is now stored in one common column
            regression_values = sample[self.target_column]
            regression_values = torch.tensor(regression_values, dtype=torch.float32)
            return file_name, features, regression_values
        
        if self.phase == "val":
            if not self.class_column:
                raise ValueError("When phase=='val', a class_column must be specified.")

            categorical_label = sample[self.class_column]
            categorical_label = torch.tensor(self.label_dict[categorical_label], dtype=torch.long)
            
            return file_name, features, categorical_label

