# -*- coding: UTF-8 -*-

import os
import h5py
from torch.optim import AdamW

import json
import random
import pickle
import json
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from transformers import AutoTokenizer, AutoModel
from trie import Trie
import logging

# # PEFT Method 
# from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
# from accelerate import Accelerator
# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
# from peft import get_peft_model, prepare_model_for_int8_training, LoraConfig
# from peft import PeftModel, PeftConfig
# from transformers import AutoModelForCausalLM, AutoTokenizer



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def check_dirs(dirs=[]):
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)
            print(f'Create directory: {dir}')
        else:
            print(f'Existed: {dir}')




def load_prefix_tree(trie_file, bos_token_id, args):
    args.logger.info(f'  load prefix tree')
    trie_dict = pd.read_pickle(trie_file)
    trie = Trie.load_from_dict(trie_dict, bos_token_id)
    args.logger.info(f'  done')
    return trie


def get_embed(file_path, mention_data, args):
    if not os.path.exists(file_path):
        with open(mention_data, encoding="utf-8") as f:
            raw_data = [json.loads(line) for line in f]
        data = [raw_data[i] for i in range(len(raw_data)) if raw_data[i]['golden'] != "NIL"] # remove NIL/OOV
        file_name = mention_data.split('/')[-1]
        
        args.roberta_tokenizer = AutoTokenizer.from_pretrained(args.simcse_model)
        args.roberta_model = AutoModel.from_pretrained(args.simcse_model).to(args.device)

        all_query_embed = []
        for d in tqdm(data, desc="getting mention embeddings", ncols=100):
            mention = d['mention']
            query = args.roberta_tokenizer(mention, padding=True, truncation=True, return_tensors="pt").to(args.device)
            with torch.no_grad():
                query_embed = args.roberta_model(**query, output_hidden_states=True, return_dict=True).pooler_output
            all_query_embed.append(query_embed)
        all_query_embed = torch.cat(all_query_embed, dim=0)
        with open(args.ment_embed_file, 'wb') as f:
            pickle.dump(all_query_embed, f)
    
    with open(file_path, 'rb') as f:
        all_embeds = pickle.load(f)

    return all_embeds


def train_configure(args):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in args.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in args.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    warmup_steps = int(args.total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=args.total_steps)
    return optimizer, scheduler



def get_logger(args):
    # 1、创建一个logger
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    # 2、创建一个handler，用于写入日志文件
    fh = logging.FileHandler(args.txtlog_dir) # ../log.log
    fh.setLevel(logging.DEBUG)
    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # 3、定义handler的输出格式（formatter）
    formatter = logging.Formatter('%(asctime)s:  %(message)s', datefmt="%m/%d %H:%M:%S")
    # 4、给handler添加formatter
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 5、给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    args.logger = logger







# def save2inference(args, save_path="output_dir"):
#     model = args.model.lm
#     model.save_pretrained(save_path)


def pure_output(text):
    text = text[0].split("\n")[0]
    text = text.split("</s>")[0]
    return text


def get_met_emb(file, args):
    import json
    import pickle
    with open(file, encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f]
    data = [raw_data[i] for i in range(len(raw_data)) if raw_data[i]['golden'] != "NIL"] # remove NIL/OOV
    file_name = file.split('/')[-1]
    # print(f'{file_name}\t\traw data num: {len(raw_data)}\t\tprocessed data num: {len(data)}')
    
    args.roberta_tokenizer = AutoTokenizer.from_pretrained(args.simcse_model)
    args.roberta_model = AutoModel.from_pretrained(args.simcse_model).to(args.device)

    all_query_embed = []
    for d in tqdm(data):
        mention = d['mention']
        query = args.roberta_tokenizer(mention, padding=True, truncation=True, return_tensors="pt").to(args.device)
        with torch.no_grad():
            query_embed = args.roberta_model(**query, output_hidden_states=True, return_dict=True).pooler_output
        all_query_embed.append(query_embed)
    all_query_embed = torch.cat(all_query_embed, dim=0)
    with open(args.ment_embed_file, 'wb') as f:
        pickle.dump(all_query_embed, f)











if __name__ == '__main__':
    pass
