import os
import sys

from pathlib import Path

import torch

from transformers.models.bert.tokenization_bert import BertTokenizer

from conf import d_model, max_len, ffn_hidden, n_heads, n_layers, drop_prob, device
from model.transformer import Transformer
from search import BeamSearch, GreedySearch


if __name__=='__main__':
    root_path = str(Path(os.path.abspath(__file__)).parent)

    # kor
    vocab_file_path = f'{root_path}/data/vocab-v1.txt'
    enc_tokenizer = BertTokenizer(vocab_file=vocab_file_path, do_lower_case=False)

    # en
    dec_vocab_file_path = f'{root_path}/data/wpm-vocab_en.txt'
    dec_tokenizer = BertTokenizer(vocab_file=dec_vocab_file_path, do_lower_case=False)

    src_pad_idx = enc_tokenizer.pad_token_id
    trg_pad_idx = dec_tokenizer.pad_token_id

    model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    d_model=d_model,
                    enc_voc_size=enc_tokenizer.vocab_size,
                    dec_voc_size=dec_tokenizer.vocab_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

    model_path = f'{root_path}/output/model_v2/checkpoints/transformer-translation-spoken.pth'

    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        start_epoch = checkpoint['epoch']
        losses = checkpoint['losses']
        global_steps = checkpoint['train_step']

        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'{model_path} loaded')

        beam_size = 5
        top_n = 3
        beam = BeamSearch(model, enc_tokenizer, dec_tokenizer, beam_size, top_n)

        input_str = '안녕하세요. 만나서 반갑습니다.'
        input_str = '저는 결혼은 자신의 삶을 경제적으로 안정되게 할 수단이라고 생각해요.'
        input_str = '은행으로 가는 버스는 어디서 타야 하나요?'
        # input_str = '저는 남는 시간에 공부해요.'

        beam_search_results = beam.translate(input_str)

        print(f'input_str : {input_str}')
        print(f'top_{top_n} -> beam search results =============================')
        for result in beam_search_results:
            print(result)

        greedy = GreedySearch(model, enc_tokenizer, dec_tokenizer)
        greedy_search_result = greedy.translate(input_str)

        print('greedy search result =============================')
        print(greedy_search_result[0])
