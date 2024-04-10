# -*- coding: UTF-8 -*-

import argparse
import os


MODEL_PATH = {
    'llama': '../data/pretrain/llama7bhf',
    'qwen': '../../_llm_weight/qwen',
    'llava': 'llava-hf/llava-1.5-7b-hf',
    'minigpt4': '../../_llm_weight/minigpt4',
    'insBlip': '../../_llm_weight/insBlip',
    'qwen_chat': '../../_llm_weight/qwen_chat',
    'qwen_chat2': '../../_llm_weight/qwen_VL_Chat',
    'qwen_chat3': '../../_llm_weight/Qwen_VL',
    'mplug': '../../_llm_weight/mPLUG-Owl2',
}

def get_args():
    args_parser = argparse.ArgumentParser(description='llama2_train_wikimel')

    # dictionary or file
    args_parser.add_argument('--dataset_dir', type=str, default='../../_ELdata/WikiMEL/')
    args_parser.add_argument('--img_file', type=str, default='../../_ELdata/WikiMEL/image_feature.h5')
    args_parser.add_argument('--example_file', type=str, default='../../_ELdata/WikiMEL/example.json')
    args_parser.add_argument('--path_image', type=str, default='../../_ELdata/ImgData/wikipedia/')
    args_parser.add_argument('--log_dir', type=str, default='./log/')
    args_parser.add_argument('--txtlog_dir', type=str, default='./log.log')
    args_parser.add_argument('--ckpt_dir', type=str, default='./checkpoint/')
    args_parser.add_argument('--trie_file', type=str, default='../../_ELdata/WikiMEL/wikimel_prefix_tree_llama.pkl')
    args_parser.add_argument('--ment_embed_file', type=str, default='../../_ELdata/WikiMEL/SimCSE_train_mention_embeddings.pkl')
    args_parser.add_argument('--weight_dir', type=str, default='../data/')

    # model
    args_parser.add_argument('--model_name', type=str, default='llama')
    args_parser.add_argument('--max_new_tokens', type=int, default=32, help='max length of gpt generation tokens')
    args_parser.add_argument('--num_beams', type=int, default=2)
    args_parser.add_argument('--use_prefix_tree', type=bool, default=False) # True False
    args_parser.add_argument('--visual_prefix_length', type=int, default=1, help='clip projected length')  # 4
    args_parser.add_argument('--dim_clip', type=int, default=512)
    args_parser.add_argument('--simcse_model', type=str, default='../data/pretrain/sup-simcse-roberta-large/') # # princeton-nlp/sup-simcse-roberta-large
    args_parser.add_argument('--dim_llm', type=int, default=4096)  # llm embedding dimension
    
    args_parser.add_argument('--add_image', type=bool, default=False)  # Image ablation
    args_parser.add_argument('--task_type', type=str, default="EL_MEL")  # EL_MEL, EL_DIV, SA_15, SA_17


    # train
    args_parser.add_argument('--train_bs', type=int, default=1)  # 1
    args_parser.add_argument('--random_seed', type=int, default=42)
    args_parser.add_argument('--lr', type=float, default=5e-6) # 1e-5 5e-6
    args_parser.add_argument('--train_epoch', type=int, default=3) # 5
    args_parser.add_argument('--weight_decay', type=float, default=0.01)
    args_parser.add_argument('--warmup_ratio', type=float, default=0.1)
    args_parser.add_argument("--max_norm", type=float, default=1, help='max norm of the gradients')
    args_parser.add_argument("--adam_epsilon", default=1e-6, type=float)

    # eval
    args_parser.add_argument('--do_eval', type=bool, default=True)
    args_parser.add_argument('--eval_bs', type=int, default=1)
    args_parser.add_argument('--eval_before_train', type=bool, default=False) # True
    args_parser.add_argument('--do_eval_steps', type=int, default=15000) # 6400
    args_parser.add_argument('--eval_stop_step', type=int, default=2000) 
    

    # test
    args_parser.add_argument('--ICL_examples_num', type=int, default=16) # 16
    args_parser.add_argument('--do_test', type=bool, default=False)

    # Gradient Accumulation
    args_parser.add_argument('--use_gradient_accumulation', type=bool, default=True)
    args_parser.add_argument('--accum_iter', type=int, default=16) # 16


    # Aspect Prediction
    args_parser.add_argument('--pre_predict', action='store_true')  # Aspect prediction
    args_parser.add_argument('--aspect_epochs', type=int, default=1)
    args_parser.add_argument('--refresh_aspect_data', type=bool, default=False)
    
    args_parser.add_argument('--aspect_batch_size', type=int, default=16)
    args_parser.add_argument('--alpha', type=float, default=0.6, nargs='?', help='display a float')
    args_parser.add_argument('--beta', type=float, default=0.6, nargs='?', help='display a float')
    args_parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    args_parser.add_argument("--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps.")
    args_parser.add_argument('--aspect_lr', type=float, default=2e-5, nargs='?', help='display a float')

    # parse
    args = args_parser.parse_args()

    # data
    if args.task_type == 'EL_MEL':
        args.data_file = {
            "train": os.path.join(args.dataset_dir, "train.json"),
            "dev": os.path.join(args.dataset_dir, "dev.json"),
            "test": os.path.join(args.dataset_dir, "test.json")
        }
    else:
        args.data_file = {
            "train": os.path.join(args.dataset_dir, "train.txt"),
            "dev": os.path.join(args.dataset_dir, "dev.txt"),
            "test": os.path.join(args.dataset_dir, "test.txt")
        }


    args.model_path = MODEL_PATH[args.model_name]
    return args











if __name__ == '__main__':
    pass
