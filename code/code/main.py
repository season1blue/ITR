# -*- coding: UTF-8 -*-
# python3 -m torch.distributed.launch --nproc_per_node=2 main.py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

from model_ import Generator

from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer



import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import params

from dataset_el import ELDataset
from dataset_sa import SADataset
from utils import check_dirs, set_seed, load_prefix_tree, get_embed, train_configure
from utils import get_logger
from transformers import AutoTokenizer
from aspect.aspect_method import aspect_method
from aspect.aspect_model import ASPModel

# 忽略not init权重的warning提示
from transformers import logging
logging.set_verbosity_error()

from evaluate import _eval4el, _eval4sa, _eval2save


# PEFT Method 
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from accelerate import Accelerator
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, LlavaForConditionalGeneration

def wrap_with_peft(args, inference=False):
    accelerator = Accelerator()
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    if args.model_name == "llama":
        args.tokenizer = LlamaTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        #  load the model in half-precision to accelerate generation and optimize memory consumption on GPU
        model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map=device_map, trust_remote_code=True)
        # freeze large language model
        # for param in model.parameters():
        #     param.requires_grad = False
        peft_config = LoraConfig(task_type="CAUSAL_LM", inference_mode=inference, r=8, lora_alpha=32, lora_dropout=0.1)
    else:
        args.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = LlavaForConditionalGeneration.from_pretrained(args.model_path, device_map=device_map, trust_remote_code=True)
        peft_config = LoraConfig(task_type="CAUSAL_LM", inference_mode=inference, r=8, lora_alpha=32, lora_dropout=0.1,target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # "gate_proj",
            # "up_proj",
            # "down_proj",
            # "lm_head",
        ])


    # model = prepare_model_for_int8_training(model)

    
    
    
    model.gradient_checkpointing_enable()  # `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
    model.enable_input_require_grads()
    
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    model = accelerator.prepare(model)

    return model


def _train(args):
    # 1.record: loss steps
    ls_sum = 0.0

    # 2.train and evaluate
    with tqdm(total=args.total_steps, ncols=100) as pbar:
        for epoch in range(args.train_epoch):
            args.epoch = epoch
            for batch_idx, batch_data in enumerate(args.train_dl):
                args.global_steps += 1

                args.model.train()
                batch_pairs, batch_targets = batch_data
                gen_ls = args.model(batch_pairs, batch_targets).loss  # 训练 HERE
                if args.use_gradient_accumulation:
                    gen_ls /= args.accum_iter
                gen_ls.backward()

                if args.use_gradient_accumulation:
                    if ((batch_idx + 1) % args.accum_iter == 0) or (batch_idx + 1 == len(args.train_dl)):
                        torch.nn.utils.clip_grad_norm_(args.model.parameters(), args.max_norm)
                        args.optimizer.step()
                        args.scheduler.step()
                        args.optimizer.zero_grad()
                else:
                    torch.nn.utils.clip_grad_norm_(args.model.parameters(), args.max_norm)
                    args.optimizer.step()
                    args.scheduler.step()
                    args.optimizer.zero_grad()

                ls_ = gen_ls.item()
                ls_sum += ls_
                # 1) record
                args.writer.add_scalar('train_loss', ls_sum/args.global_steps, args.global_steps)
                pbar.set_description(f'Traing Epoch: {epoch}, steps: {args.global_steps}, loss: {ls_:.4f}')
                pbar.update(1)
                # 2) evaluate and save
                if args.do_eval and not args.global_steps % args.do_eval_steps:
                    _eval2save(args)

def test(args):
    # if args.task_type == "EL_MEL":
    #     test_ds = ELDataset(args.data_file['test'], args.tokenizer, args, **args.kwargs_ds)
    # else:
    #     test_ds = SADataset(args.data_file['test'], args.tokenizer, args, **args.kwargs_ds)
    # test_dl = DataLoader(test_ds, batch_size=args.eval_bs, collate_fn=test_ds.collate_fn, shuffle=False)


    args.logger.info('load linear')
    checkpoint_file = f'{args.ckpt_dir}{args.model_name}_linear_{args.visual_prefix_length}token_{args.ICL_examples_num}examples.pkl'
    print(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    args.model.linear.load_state_dict(checkpoint)

    precision, recall, f1 = _eval4sa(args, args.eval_dl, mode="test")
    print(precision, recall, f1)

def _main(args):

    # 1.check dir
    check_dirs(dirs=[args.dataset_dir, args.log_dir, args.ckpt_dir])

    # 2.random seed
    set_seed(args.random_seed)

    # 3.model and tokenizer
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model, mention embeddings for calculating similarity
    args.roberta_tokenizer = AutoTokenizer.from_pretrained(args.simcse_model)
    args.roberta_model = AutoModel.from_pretrained(args.simcse_model).to(args.device)

    # apsect predictor init and generate aspect_span
    aspect_predictor = aspect_method(args=args)
    train_aspect = aspect_predictor.predict("train")
    dev_aspect = aspect_predictor.predict("dev")
    




    # 4.data
    # train dataset for calculating ICL similarity
    if args.task_type == "EL_MEL":
        args.kwargs_ds = {'ICL_examples_num': args.ICL_examples_num, 'img_file': args.img_file, 'device': args.device,
                        'roberta_tokenizer': args.roberta_tokenizer, 'roberta_model': args.roberta_model, 'path_image':args.path_image, 'dataset_dir':args.dataset_dir}
        args.ICL_ds = ELDataset(args.data_file['train'], tokenizer=None, args=args, **args.kwargs_ds)
        train_ds = ELDataset(args.data_file['train'], tokenizer=True, args=args, train_flag=True, **args.kwargs_ds)  # train_flag: for exclude same training example
        dev_ds = ELDataset(args.data_file['dev'], tokenizer=True, args=args, **args.kwargs_ds)
    else:
        args.kwargs_ds = {'ICL_examples_num': args.ICL_examples_num, 'img_file': args.img_file, 'device': args.device,
                        'roberta_tokenizer': args.roberta_tokenizer, 'roberta_model': args.roberta_model, 'path_image':args.path_image, 'dataset_dir':args.dataset_dir, 'example_file':args.example_file}
        # args.ICL_ds = SADataset(args.data_file['train'], tokenizer=None, args=args, **args.kwargs_ds)
        train_ds = SADataset(args.data_file['train'], aspect=train_aspect, tokenizer=True, args=args, train_flag=True, **args.kwargs_ds)  # train_flag: for exclude same training example
        dev_ds = SADataset(args.data_file['dev'], aspect=dev_aspect, tokenizer=True, args=args, **args.kwargs_ds)


    args.logger.info(f'train data num: {len(train_ds)}  dev data num: {len(dev_ds)}')
    args.train_dl = DataLoader(dataset=train_ds, batch_size=args.train_bs, collate_fn=train_ds.collate_fn, shuffle=True)  # args.train_bs
    args.eval_dl = DataLoader(dataset=dev_ds, batch_size=args.eval_bs, collate_fn=dev_ds.collate_fn, shuffle=False)  # args.eval_bs
    args.total_steps = args.train_epoch * len(args.train_dl)

    # prefix tree
    args.trie = load_prefix_tree(args.trie_file, args.tokenizer.bos_token_id, args) if args.use_prefix_tree else None

    ########  llm load   ########
    lm = wrap_with_peft(args, inference=False)

    args.dim_embedding = args.dim_llm 
    kwargs_model = {'dim_clip': args.dim_clip, 'dim_embedding': args.dim_embedding,
                    'visual_prefix_length': args.visual_prefix_length, 'device': args.device, 'add_image': args.add_image}
    args.model = Generator(args=args, lm=lm, tokenizer=args.tokenizer, inference=False, **kwargs_model).to(args.device)

    args.optimizer, args.scheduler = train_configure(args)

    args.writer = SummaryWriter(args.log_dir)
    args.global_steps = 0
    args.best_f1, args.best_precision, args.best_recall = 0, 0, 0
    args.best_f1, args.best_precision, args.best_recall = _eval2save(args) if args.eval_before_train else 0, 0, 0

    if args.do_test:
        test(args)
    else:
        _train(args)
        # # 6.inference test
        # if args.do_test: test(args)

        # Final Eval (New Add)
        _eval2save(args)

    




if __name__ == '__main__':
    args = params.get_args()
    get_logger(args) 

    try:
        _main(args)
    except Exception as e:
        args.logger.error("Error", exc_info=True)


