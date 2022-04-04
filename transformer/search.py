import sys
from abc import *

import torch

from transformers.models.bert.tokenization_bert import BertTokenizer

from conf import max_len, device
from model.transformer import Transformer


class Search(metaclass=ABCMeta):
    def __init__(self, model: Transformer, enc_tokenizer: BertTokenizer, dec_tokenizer: BertTokenizer) -> None:
        self.model = model
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer

    @abstractmethod
    def search(self, text: str) -> list:
        pass

    def translate(self, text: str) -> list:
        search_result = self.search(text)

        return [ self.dec_tokenizer.decode(result, skip_special_tokens=True) for result in search_result ]


class GreedySearch(Search):
    def __init__(self, model: Transformer, enc_tokenizer: BertTokenizer, dec_tokenizer: BertTokenizer) -> None:
        super(GreedySearch, self).__init__(model, enc_tokenizer, dec_tokenizer)

    def search(self, text: str) -> list:
        self.model.eval()
        with torch.no_grad():
            enc_str = self.enc_tokenizer.encode(text)
            src = torch.tensor([enc_str]).to(device)

            trg = torch.ones(1, 1).fill_(self.dec_tokenizer.cls_token_id).type_as(src)

            for i in range(0, max_len):
                lm_logits, _ = self.model(src, trg)

                prob = lm_logits[:, -1]
                _, next_word = torch.max(prob, dim=1)

                if next_word.data[0] == self.dec_tokenizer.sep_token_id:
                    break

                trg = torch.cat((trg[0, :i+1], next_word))
                trg = trg.unsqueeze(0)
            
            return trg


class BeamSearch(Search):
    def __init__(self, model: Transformer, enc_tokenizer: BertTokenizer, dec_tokenizer: BertTokenizer, beam_size: int = 5, top_n: int = 5) -> None:
        super(BeamSearch, self).__init__(model, enc_tokenizer, dec_tokenizer)
        self.beam_size = beam_size
        self.top_n = top_n

    def search(self, text: str) -> list:
        self.model.eval()
        with torch.no_grad():
            softmax = torch.nn.Softmax(dim=1)

            enc_str = self.enc_tokenizer.encode(text)
            src = torch.tensor([enc_str]).to(device)

            trg = torch.ones(1, 1).fill_(self.dec_tokenizer.cls_token_id).type_as(src)
        
            score_dict = {}

            alpha = 0.95
            for i in range(max_len - 1):
                if i == 0:
                    lm_logits, _ = self.model(src, trg)

                    prob = softmax(lm_logits[:, -1])
                    sorted_prob, indices = torch.sort(prob, dim=1, descending=True)
                    prev_probs = sorted_prob[:, :self.beam_size]
                    next_indices = indices[:, :self.beam_size]

                    trg = trg.repeat(1, self.beam_size)
                    trg = torch.stack((trg, next_indices), dim=2).squeeze(0)
                else:
                    # remove endswith [SEP]
                    input_trg = None
                    for trg_idx, trg_element in enumerate(trg):
                        if trg_element[-1] == self.dec_tokenizer.sep_token_id:
                            # shape을 맞춰 주기 위해 trg_element에 [SEP] 추가
                            score_dict[torch.cat((trg_element, torch.tensor([self.dec_tokenizer.sep_token_id]).to(device)))] = prev_probs[0][trg_idx] * alpha
                        else:
                            if input_trg is not None:
                                input_trg = torch.cat((input_trg, trg_element.unsqueeze(0)), dim=0)
                            else:
                                input_trg = torch.stack([trg_element])

                    # all trg endswith [SEP]
                    if input_trg is None:
                        break

                    trg_length = input_trg.size(0)
                    input_src = src.repeat(trg_length, 1)
                    lm_logits, _ = self.model(input_src, input_trg)

                    prob = softmax(lm_logits[:, -1])
                    sorted_prob, indices = torch.sort(prob, dim=1, descending=True)

                    top_k_sorted_prob = sorted_prob[:, :self.beam_size]
                    next_indices = indices[:, :self.beam_size]

                    for j in range(0, trg_length):
                        # top_k sequence
                        for l in range(0, self.beam_size):
                            # candidate_trg = torch.cat((trg[j], next_indices[j][l].unsqueeze(0)))
                            candidate_trg = torch.cat((input_trg[j], next_indices[j][l].unsqueeze(0)))
                            score_dict[candidate_trg] = prev_probs[0][j] * top_k_sorted_prob[j][l]

                            sorted_score_dict = sorted(score_dict.items(), key=lambda item: item[1].item(), reverse=True)[:self.beam_size]
                
                    trg = torch.stack([element[0] for element in sorted_score_dict])
                    prev_probs = torch.stack([element[1] for element in sorted_score_dict]).unsqueeze(0)
                    
                    score_dict.clear()

        return trg.tolist()[:self.top_n]