import os
import sys

from pathlib import Path

import time

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
import math

from transformers.models.bert.tokenization_bert import BertTokenizer

from conf import d_model, max_len, ffn_hidden, n_heads, n_layers, drop_prob, clip, device
from model.transformer import Transformer
from data import TranslationDataset

root_path = str(Path(os.path.abspath(__file__)).parent)
writer = SummaryWriter(f'{root_path}/runs/translation_v1.0')


class TranslationTrainer:

    def __init__(self, dataset: TranslationDataset, src_tokenizer: BertTokenizer, trg_tokenizer: BertTokenizer, model: Transformer, 
                max_length: int, device: str, model_name: str, checkpoint_path: str, batch_size: int) -> None:
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.model = model
        self.max_length = max_length
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.src_vocab_size = src_tokenizer.vocab_size
        self.trg_vocab_size = trg_tokenizer.vocab_size
        self.batch_size = batch_size

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    def trans_collate_fn(self, samples: list) -> dict:
        src_str = []
        trg_str = []
        src_str.append([sample['src_str'] for sample in samples])
        # padding_value 변경 필요
        src = pad_sequence([sample['src'] for sample in samples], batch_first=True, padding_value=0)
        trg = pad_sequence([sample['trg'] for sample in samples], batch_first=True, padding_value=0)
        trg_str.append([sample['trg_str'] for sample in samples])

        return {
            "src_str": src_str,
            "src": src,
            "trg_str": trg_str,
            "trg": trg
        }

    def build_dataloaders(self, train_test_split: int=0.2, train_shuffle: bool=True, eval_shuffle: bool=True) -> tuple:
        dataset_len = len(self.dataset)
        eval_len = int(dataset_len * train_test_split)
        train_len = dataset_len - eval_len
        train_dataset, eval_dataset = random_split(self.dataset, (train_len, eval_len))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=train_shuffle, collate_fn=self.trans_collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=eval_shuffle , collate_fn=self.trans_collate_fn)

        return train_loader, eval_loader

    def train(self, epochs: int, train_dataset: DataLoader, eval_dataset: DataLoader, optimizer: Optimizer, scheduler) -> None:
        self.model.train()
        total_loss = 0.0
        global_steps = 0
        start_time = time.time()
        losses = {}
        best_val_loss = float("inf")
        best_model = None
        start_epoch = 0
        start_step = 0
        train_dataset_length = len(train_dataset)

        self.model.to(self.device)
        if os.path.isfile(f'{self.checkpoint_path}/{self.model_name}.pth'):
            checkpoint = torch.load(f'{self.checkpoint_path}/{self.model_name}.pth', map_location=self.device)
            start_epoch = checkpoint['epoch']
            losses = checkpoint['losses']
            global_steps = checkpoint['train_step']
            start_step = global_steps if start_epoch == 0 else (global_steps % train_dataset_length) + 1

            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logging_loss = 0.0
        logging_step = 0

        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()

            pb = tqdm(enumerate(train_dataset),
                      desc=f'Epoch-{epoch} Iterator',
                      total=train_dataset_length,
                      bar_format='{l_bar}{bar:10}{r_bar}')
            pb.update(start_step)

            for i, data in pb:

                if i < start_step:
                    continue

                src = data['src']
                trg = data['trg']

                optimizer.zero_grad()
                output, _ = self.model(src, trg[:, :-1])
                output_reshape = output.contiguous().view(-1, output.shape[-1])
                trg = trg[:, 1:].contiguous().view(-1)

                loss = self.criterion(output_reshape, trg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                losses[global_steps] = loss.item()
                total_loss += loss.item()
                logging_loss += loss.item()
                log_interval = 1
                # save_interval = 500
                save_interval = 5000

                global_steps += 1

                if i % log_interval == 0 and i > 0:
                    cur_loss = total_loss / log_interval
                    elapsed = time.time() - start_time

                    pb.set_postfix_str('| epoch {:3d} | {:5d}/{:5d} batches | '
                                        'lr {:02.2f} | ms/batch {:5.2f} | '
                                        'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, i, len(train_dataset), scheduler.get_last_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))

                    total_loss = 0
                    start_time = time.time()

                    if i % save_interval == 0:
                        val_loss = self.evaluate(eval_dataset)
                        self.model.train()
                        print()
                        print('-' * 89)
                        print('| epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                        val_loss, math.exp(val_loss)))
                        print('-' * 89)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.save(epoch, self.model, optimizer, losses, global_steps)

                # log tensorboard
                if logging_step % 1000 == 0 and logging_step > 0:
                    writer.add_scalar('training loss',
                            logging_loss / 1000,
                            # i + (epoch * len(pb)))
                            global_steps)
                    logging_loss = 0.0
                    logging_step = 0
                else:
                    logging_step += 1

            scheduler.step()
    
    def evaluate(self, dataset: DataLoader) -> float:
        self.model.eval()  # 평가 모드를 시작합니다.
        total_loss = 0.0

        self.model.to(self.device)
        with torch.no_grad():
            for i, data in enumerate(dataset):
                src = data['src']
                trg = data['trg']

                output, _ = self.model(src, trg[:, :-1])
                output_reshape = output.contiguous().view(-1, output.shape[-1])
                trg = trg[:, 1:].contiguous().view(-1)

                loss = self.criterion(output_reshape, trg)
                total_loss += loss.item()

        return total_loss / (len(dataset) - 1)


    def save(self, epoch: int, model: Transformer, optimizer: Optimizer, losses: float, train_step: int):
        print('save best model.')
        torch.save({
            'epoch': epoch,  # 현재 학습 epoch
            'model_state_dict': model.state_dict(),  # 모델 저장
            'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
            'losses': losses,  # Loss 저장
            'train_step': train_step,  # 현재 진행한 학습
            }, f'{self.checkpoint_path}/{self.model_name}.pth')


if __name__ == '__main__':
    # kor
    vocab_file_path = f'{root_path}/data/vocab-v1.txt'
    enc_tokenizer = BertTokenizer(vocab_file=vocab_file_path, do_lower_case=False)

    # en
    dec_vocab_file_path = f'{root_path}/data/wpm-vocab_en.txt'
    dec_tokenizer = BertTokenizer(vocab_file=dec_vocab_file_path, do_lower_case=False)

    src_pad_idx = enc_tokenizer.pad_token_id
    trg_pad_idx = dec_tokenizer.pad_token_id

    torch.manual_seed(10)
    checkpoint_path = f'{root_path}/output/model_v2/checkpoints'

    # model setting
    model_name = 'transformer-translation-spoken'

    # hyper parameter
    epochs = 50
    batch_size = 4
    learning_rate = 0.5

    data_path = f'{root_path}/data/kor_eng_transform'
    dataset = TranslationDataset(src_tokenizer=enc_tokenizer, trg_tokenizer=dec_tokenizer, src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx, file_path=data_path, max_length=max_len, device=device)

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

    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1.0, gamma=0.95)

    trainer = TranslationTrainer(dataset, enc_tokenizer, dec_tokenizer, model, max_len, device, model_name, checkpoint_path, batch_size)
    train_dataloader, eval_dataloader = trainer.build_dataloaders(train_test_split=0.2)

    trainer.train(epochs, train_dataloader, eval_dataloader, optimizer, scheduler)