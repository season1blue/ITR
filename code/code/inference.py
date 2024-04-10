# -*- coding: UTF-8 -*-


import os

import pickle


os.environ['CUDA_VISIBLE_DEVICES'] = "5"


from model_ import Generator



from torch.utils.tensorboard import SummaryWriter

from transformers import AutoModel, AutoTokenizer


import random


import torch

from torch.utils.data import DataLoader

from tqdm import tqdm
import params

from utils import check_dirs, set_seed, ELDataset, calc_acc, load_prefix_tree, get_embed, train_configure
from main import wrap_with_peft
from peft import PeftModel, PeftConfig

from accelerate import Accelerator

from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM

from peft import get_peft_model, prepare_model_for_int8_training, LoraConfig


def wrap_with_peft_inference(args, inference=True):

    peft_model_id = "output_dir"

    peft_config = PeftConfig.from_pretrained(peft_model_id)


    #  load the model in half-precision to accelerate generation and optimize memory consumption on GPU

    model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)


    # for param in model.parameters():

    #     param.requires_grad = False

    # # model = prepare_model_for_int8_training(model)
    

    # model.gradient_checkpointing_enable()  # `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...

    # model.enable_input_require_grads()
    

    model = PeftModel.from_pretrained(model, peft_model_id)


    model.eval()    

    return model 




def test(args):
    test_ds = ELDataset(args.data_file['dev'], args.tokenizer, **args.kwargs_ds)

    test_dl = DataLoader(test_ds, batch_size=args.eval_bs, collate_fn=test_ds.collate_fn, shuffle=False)


    print('\nload linear')

    checkpoint_file = f'{args.ckpt_dir}{args.model_name}_linear_{args.visual_prefix_length}token_{args.ICL_examples_num}examples.pkl'

    checkpoint = torch.load(checkpoint_file)

    args.model.linear.load_state_dict(checkpoint)


    acc = _eval(args, test_dl)



def _eval(args, dl):

    print('Test...')

    # record

    eval_steps = len(dl)

    input_texts, predictions, targets, goldens = [], [], [], []

    # evaluate

    args.model.eval()

    with torch.no_grad():

        with tqdm(total=eval_steps) as pbar:

            for step, batch_data in enumerate(dl):

                pbar.set_description(f'eval steps: {step}')

                pbar.update(1)


                batch_pairs, batch_targets, batch_golden = batch_data

                features = {

                    "batch_pairs": batch_pairs,

                    "num_beams": args.num_beams,

                    "num_return_sequences": 1,

                    "max_new_tokens": args.max_new_tokens,

                }

                if args.use_prefix_tree:

                    features['prefix_allowed_tokens_fn'] = lambda batch_id, sent: args.trie.get(sent.tolist())

                generated = args.model.generate(**features, max_new_tokens=10)

                batch_preds = args.tokenizer.batch_decode(generated.detach().cpu().numpy(), skip_special_tokens=True)

                predictions.extend(batch_preds)

                targets.extend(batch_targets)

                goldens.extend(batch_golden)


                # save prediction

                for pair in batch_pairs:

                    input_text = ''.join([t for _, t in pair])

                    input_texts.append(input_text)


                # log prediction

                if not step % 30:

                    i = random.randint(0, len(batch_targets) - 1)

                    input_text = ''.join([t for _, t in batch_pairs[i][-4:]])

                    print(f'\ninput_text:\n{input_text}')

                    print(f'\nresult: {batch_targets[i]==batch_preds[i].strip(" ")}\t\ttarget: {batch_targets[i]}\t\tpred: {batch_preds[i]}')

    acc = calc_acc(predictions, targets)


    assert len(input_texts) == len(targets) and len(targets) == len(predictions) and len(predictions) == len(goldens)

    string_list = []

    result = []

    for i in range(len(input_texts)):

        string_list.append(f'\ninput_text:\n{input_texts[i]}')
        string_list.append(

            f'result: {targets[i] == predictions[i].strip(" ")}\t\ttarget: {targets[i]}\t\tpred: {predictions[i]}')

        result.append(

            (targets[i].strip().lower() == predictions[i].strip().lower(), goldens[i], targets[i], predictions[i]))


        with open(f"./wikidata_{args.model_name}_result.pkl", 'wb') as f:

            pickle.dump(result, f)


    string_list.append(f'\nacc: {acc} %')

    with open(f'wikiMEL_img_{args.model_name}_result.txt', 'w') as file:

        file.write('\n'.join(string_list))

    return acc



def _eval2save(args):

    acc = _eval(args, args.eval_dl)

    args.writer.add_scalar('eval_acc', acc, args.global_steps)

    # judge to save

    if acc >= args.best_eval_acc:

        print(f'\nNew best model, new acc {acc:.4f} % >= previous acc {args.best_eval_acc:.4f} %')

        args.best_eval_acc = acc

        checkpoint_file = f'{args.ckpt_dir}{args.model_name}_linear_{args.visual_prefix_length}token_{args.ICL_examples_num}examples.pkl'  # only save 1 checkpoint

        torch.save(args.model.linear.state_dict(), checkpoint_file)

        print(f'\nSave to {checkpoint_file}')


    elif acc < args.best_eval_acc:

        print(f'\ndo not save, best acc: {args.best_eval_acc:.4f}')




def _main(args):

    # 1.check dir

    check_dirs(dirs=[args.dataset_dir, args.log_dir, args.ckpt_dir])


    # 2.random seed
    set_seed(args.random_seed)


    # 3.model and tokenizer

    from transformers import LlamaTokenizer, LlamaForCausalLM
    

    args.tokenizer = LlamaTokenizer.from_pretrained(args.model_path, legacy=False)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # #  load the model in half-precision to accelerate generation and optimize memory consumption on GPU

    # lm = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)

    # # freeze large language model

    # print(f'\nFreeze LLM {args.model_name}\n')

    # for param in lm.parameters():

    #     param.requires_grad = False

    lm = wrap_with_peft_inference(args, inference=True)


    args.dim_embedding = 4096 # lm.config.hidden_size

    kwargs_model = {'dim_clip': args.dim_clip, 'dim_embedding': args.dim_embedding,

                    'visual_prefix_length': args.visual_prefix_length, 'device': args.device}

    args.model = Generator(lm=lm, tokenizer=args.tokenizer, inference=True, **kwargs_model).to(args.device)


    # model, mention embeddings for calculating similarity

    args.train_embed = get_embed(args.ment_embed_file, mention_data = args.data_file['train'], args=args)

    args.roberta_tokenizer = AutoTokenizer.from_pretrained(args.simcse_model)

    args.roberta_model = AutoModel.from_pretrained(args.simcse_model).to(args.device)


    # 4.data

    # train dataset for calculating ICL similarity

    args.ICL_ds = ELDataset(args.data_file['train'], tokenizer=None, img_file=args.img_file)


    args.kwargs_ds = {'train_ds': args.ICL_ds, 'ICL_examples_num': args.ICL_examples_num, 'img_file': args.img_file, 'device': args.device,

                      'train_embed': args.train_embed, 'roberta_tokenizer': args.roberta_tokenizer, 'roberta_model': args.roberta_model}


    # prefix tree

    args.trie = load_prefix_tree(args.trie_file, args.tokenizer.bos_token_id) if args.use_prefix_tree else None



    # Inference test

    if args.do_test: test(args)





if __name__ == '__main__':

    # 1.args
    args = params.get_args()
    print(args)


    # 2._mian
    _main(args)


