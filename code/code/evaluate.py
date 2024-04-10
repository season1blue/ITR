import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import params
from dataset_el import ELDataset
from dataset_sa import SADataset
from utils import pure_output 
import re





def calc_acc(predictions, targets, args):
    assert len(predictions) == len(targets)
    hits = [predictions[i] == targets[i] for i in range(len(targets))]
    acc = 100.0 * sum(hits) / len(hits)
    return acc

def match_result(text, predict=True):
    aspect_pattern = r"text\]\[(.+?)\]"
    sentiment_pattern = r"emotion\]\[(.+?)\]"

    aspect = re.findall(aspect_pattern, text)
    sentiment = re.findall(sentiment_pattern, text)
    # if predict:
    #     print(text)
    #     print(aspect, sentiment)
    #     print("-----")
    if len(aspect) == len(sentiment):
        pairs = [(a,s) for a, s in zip(aspect, sentiment)]
        num = len(pairs)
    elif len(aspect) < len(sentiment):
        pairs = [(a, sentiment[index]) for index,a in enumerate(aspect)]
        num = len(sentiment)
    else:
        pairs = [(aspect[index], s) for index, s in enumerate(sentiment)]
        num = len(aspect)
    return set(pairs), num

def calc_f1(predictions, targets, mode="train"):
    predict_num, target_num, correct_num = 0, 0, 0
    for prediction, target in zip(predictions, targets):
        if mode == "test":
            print(prediction)
            print(target)
        predict_pairs, p_num = match_result(prediction, True)
        target_pairs, t_num  = match_result(target, False)
        c_num = len(predict_pairs & target_pairs)
        predict_num += p_num
        target_num += t_num
        correct_num += c_num


    precision = correct_num / predict_num if predict_num != 0 else 0
    recall = correct_num / target_num if target_num != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision != 0 or recall != 0 else 0

    return precision * 100, recall * 100, f1 * 100




def _eval4el(args, dl):
    # record
    eval_steps = len(dl)
    predictions, targets = [], []

    args.model.eval()

    with torch.no_grad():
        for step, batch_data in tqdm(enumerate(dl), desc="Eval", ncols=80, total=eval_steps):
            # if step>500:break
            batch_pairs, batch_targets = batch_data
            features = {
                "batch_pairs": batch_pairs,
                "num_beams": args.num_beams,
                "num_return_sequences": 1,
                "max_new_tokens": args.max_new_tokens,
            }
            if args.use_prefix_tree:
                features['prefix_allowed_tokens_fn'] = lambda batch_id, sent: args.trie.get(sent.tolist())
            
            generated = args.model.generate(**features)
            batch_preds = args.tokenizer.batch_decode(generated, skip_special_tokens=True)
            # batch_preds = args.tokenizer.batch_decode(torch.argmax(generated.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)

            predictions.extend([pure_output(batch_preds)])  # 
            targets.extend(batch_targets)  # 
            # print([pure_output(batch_preds)] == batch_targets, [pure_output(batch_preds)], batch_targets) # ['Josep Llunas i Pujals\n[Question]What does Josep Llunas i Pujals mentioned in the text refer to?\n']
            # print(batch_targets) # ['Josep Llunas i Pujals']
            if not step % 30:
                i = random.randint(0, len(batch_targets) - 1)
                input_text = ''.join([t for _, t in batch_pairs[i][-4:]])
    precision, recall, f1 = calc_f1(predictions, targets)
    return precision, recall, f1

def _eval4sa(args, dl, mode="train"):
    # record
    eval_steps = len(dl)
    predictions, targets = [], []

    args.model.eval()
    with torch.no_grad():
        for step, batch_data in tqdm(enumerate(dl), desc="Eval", ncols=80, total=eval_steps):
            if step > args.eval_stop_step:break
            batch_pairs, batch_targets = batch_data
            features = {
                "batch_pairs": batch_pairs,
                "num_beams": args.num_beams,
                "num_return_sequences": 1,
                "max_new_tokens": args.max_new_tokens,
            }
            if args.use_prefix_tree:
                features['prefix_allowed_tokens_fn'] = lambda batch_id, sent: args.trie.get(sent.tolist())
            
            generated = args.model.generate(**features)
            batch_preds = args.tokenizer.batch_decode(generated, skip_special_tokens=True)
            # batch_preds = args.tokenizer.batch_decode(torch.argmax(generated.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)

            predictions.extend([pure_output(batch_preds)])  # 
            targets.extend(batch_targets)  # 
            # print([pure_output(batch_preds)] == batch_targets, [pure_output(batch_preds)], batch_targets) # ['Josep Llunas i Pujals\n[Question]What does Josep Llunas i Pujals mentioned in the text refer to?\n']
            # print(batch_targets) # ['Josep Llunas i Pujals']
            if not step % 30:
                i = random.randint(0, len(batch_targets) - 1)
                input_text = ''.join([t for _, t in batch_pairs[i][-4:]])
    precision, recall, f1 = calc_f1(predictions, targets, mode)
    return precision, recall, f1



def _eval2save(args):
    if args.task_type == "EL_MEL":
        precision, recall, f1 = _eval4el(args, args.eval_dl)
    else:
        precision, recall, f1 = _eval4sa(args, args.eval_dl)

    args.writer.add_scalar('eval_f1', f1, args.global_steps)
    # judge to save
    if f1 >= args.best_f1:
        args.logger.info(f'New best model, new f1 {f1:.4f} % >= previous f1 {args.best_f1:.4f} %')
        args.best_f1, args.best_precision, args.best_recall = f1, precision, recall
        checkpoint_file = f'{args.ckpt_dir}{args.model_name}_linear_{args.visual_prefix_length}token_{args.ICL_examples_num}examples.pkl'  # only save 1 checkpoint
        torch.save(args.model.linear.state_dict(), checkpoint_file)
        
        args.logger.info(f'Save to {checkpoint_file}')

    elif f1 < args.best_f1:
        args.logger.info(f'do not save, best f1: {args.best_f1:.4f}')
    args.logger.info(f'Step:{args.global_steps}, Best: f1:{args.best_f1:.3f}, p:{args.best_precision:.3f},r:{args.best_recall:.3f}, Curr: f1:{f1:.4f}, p:{precision:.3f},r:{recall:.3f} ')
    return args.best_f1, args.best_precision, args.best_recall
