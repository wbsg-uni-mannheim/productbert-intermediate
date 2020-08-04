
from torch.utils.data import Dataset
import pandas as pd

class BertDataset(Dataset):

    def __init__(self, filename):

        # Store the contents of the file in a pandas dataframe
        self.data = self._convert_to_tensor(filename)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        record = self.data.loc[index]
        input_id = record['input_ids']
        token_id = record['token_type_ids']
        attn_mask = record['attention_mask']
        label = record['label']

        return input_id, token_id, attn_mask, label

    def _convert_to_tensor(self, filename):
        data = pd.read_pickle(filename, compression='gzip')
        data = data[['pair_id', 'input_ids', 'token_type_ids', 'attention_mask', 'label']]
        return data

class BertDatasetMLM(Dataset):

    def __init__(self, filename):

        # Store the contents of the file in a pandas dataframe
        self.data = self._convert_to_tensor(filename)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        record = self.data.loc[index]
        input_id = record['input_ids']
        token_id = record['token_type_ids']
        attn_mask = record['attention_mask']
        label = record['label']
        mlm_labels = record['mlm_labels']

        return input_id, token_id, attn_mask, label, mlm_labels

    def _convert_to_tensor(self, filename):
        data = pd.read_pickle(filename, compression='gzip')
        data = data[['pair_id', 'input_ids', 'token_type_ids', 'attention_mask', 'label', 'mlm_labels']]
        return data
        