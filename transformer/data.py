import os

import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from transformers.models.bert.tokenization_bert import BertTokenizer


def load_translation_excel(file_path: str) -> zip:

    train_excel_files = os.listdir(file_path)

    ko_text_list = []
    en_text_list = []

    for file_name in tqdm(train_excel_files[:2]):
        if file_name.endswith('.xlsx'):
            df_temp = pd.read_excel(os.path.join(file_path, file_name))
            ko_text_list.extend(df_temp['원문'].to_list())
            en_text_list.extend(df_temp['번역문'].to_list())

    excel_datasets = zip(ko_text_list, en_text_list)

    return excel_datasets


class TranslationDataset(Dataset):

    def __init__(self, src_tokenizer: BertTokenizer, trg_tokenizer: BertTokenizer, src_pad_idx: int, 
                trg_pad_idx:int, file_path: str, max_length: str, device: str) -> None:
        excel_datasets = load_translation_excel(file_path)

        self.docs = []
        for i, excel_data in tqdm(enumerate(excel_datasets)):
            src = src_tokenizer.encode(excel_data[0], max_length=max_length, truncation=True)
            src = torch.tensor(src).to(device)
            
            trg = trg_tokenizer.encode(excel_data[1], max_length=max_length, truncation=True)
            trg = torch.tensor(trg).to(device)

            doc = {
                'src_str': src_tokenizer.convert_ids_to_tokens(src),
                'src': src,
                'trg_str': trg_tokenizer.convert_ids_to_tokens(trg),
                'trg': trg
            }

            self.docs.append(doc)
            
            # for test
            # if i > 3000:
            #     break

    def __len__(self) -> int:
        return len(self.docs)

    def __getitem__(self, idx) -> dict:
        return self.docs[idx]