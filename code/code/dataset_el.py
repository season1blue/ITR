from torch.utils.data import Dataset
import json
import random
import pickle
import json
import numpy as np
import pandas as pd
import torch
import os
import h5py
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

class ELDataset(Dataset):

    def __init__(self, file, tokenizer, args, **kwargs):
        self.file = file
        self.data = self._get_data(file)

        self.tokenizer = tokenizer
        self.kwargs = kwargs

        refresh = False
        if tokenizer: # ICL dataset does not need tokenizer and examples
            if not os.path.exists(self.kwargs['img_file']) or refresh:   # Image file exist, handle image feature
                self._get_img_feat()
            self._add_img_feat() # convert image url to clip output feature
            self._get_examples(args) # get ICL examples from training set


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def _get_data(self, file):
        # load file and fiter NIL/OOV
        with open(file, encoding="utf-8") as f:
            raw_data = [json.loads(line) for line in f]
        data = [raw_data[i] for i in range(len(raw_data)) if raw_data[i]['golden'] != "NIL"] # remove NIL/OOV
        file_name = file.split('/')[-1]
        print(f'{file_name}\t\traw data num: {len(raw_data)}\t\tprocessed data num: {len(data)}')
        return data

    def _get_img_feat(self):
        import clip
        import os
        from PIL import Image
        device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training
        Image.MAX_IMAGE_PIXELS = 2300000000  # 更改阈值像素上限

        img_list = os.listdir(self.kwargs['path_image'])
        image_embeddings = torch.FloatTensor(len(img_list), 512)
        model.eval()
        model.to(device)

        image_file_path = os.path.join(self.kwargs["dataset_dir"], "image_feature.h5")
        if os.path.exists(image_file_path):
            os.remove(image_file_path)
        image_file = h5py.File(os.path.join(self.kwargs["dataset_dir"], "image_feature.h5"), 'w')

        for index, item in tqdm(enumerate(img_list), total=len(img_list), desc="extracting image feature", ncols=100):
            img_path = os.path.join(self.kwargs['path_image'], item)

            try:
                image = Image.open(img_path)
                image = preprocess(image)
                image = image.unsqueeze(0).to(device)

                with torch.no_grad():
                    image_feature = model.encode_image(image).to(device)
                image_file[item] = image_feature.cpu().numpy()
            except Exception as e :
                args.logger.info(e)
                args.logger.info(item)
                exit()
        json.dump(img_list, open(os.path.join(self.kwargs["dataset_dir"], "img_list.json"), 'w'), indent=2)
        image_file.close()
        # image_file = h5py.File(os.path.join(args.dataset_dir, "image_feature.h5"), 'w')
        # image_file.create_dataset("features", data=image_embeddings.numpy())



    def _add_img_feat(self):
        # convert image url to clip output feature
        print("adding image feature into self.data")
        with h5py.File(self.kwargs['img_file'], 'r') as h:
            for tmpDict in self.data:
                if 'img_name' in tmpDict: # wikimel dataset
                    img_url = tmpDict['img_name']
                else: # wikidiverse dataset
                    img_url = tmpDict["img_url"]
                
                try:
                    tmpDict["image"] = h[img_url][()]  # add image attribution to self.data
                except:
                    print(f"{img_url} is missing")
                    exit()
                    # raise Exception(f'\ncan not load {img_url} from {self.kwargs["img_feat"]}')

    # Finding candidate entity
    def _get_examples(self, args):
        args.logger.info(f'  Retrieve {self.file} ICL examples')
        # dataset_dir = self.kwargs["dataset_dir"]
        # example_file_path = os.path.join(dataset_dir, f"examples_{self.file}.pkl")
        example_file_path = self.file.split(".json")[0] + "_examples.json"
        args.logger.info(example_file_path)
        refresh_example = False

        if os.path.exists(example_file_path) and not refresh_example:  # 存在example 且 不需要重新生成
            prefix_items_list = json.load(open(example_file_path, "r"))
            for index, tmpDict in tqdm(enumerate(self.data), desc="loading examples from examples.json", ncols=100):
                tmpDict['examples'] = prefix_items_list[index]
            # self.data = pickle.load(open(example_file_path, 'rb'))
            # for tmpDict in self.data:
            #     args.logger.info(tmpDict['examples'][0].keys)
            #     exit()
        else:
            prefix_items_list = []
            for tmpDict in tqdm(self.data, desc="getting examples", ncols=100):
                mention = tmpDict['mention']
                text = tmpDict['text']

                query = self.kwargs['roberta_tokenizer'](mention, padding=True, truncation=True, return_tensors="pt").to(self.kwargs['device'])
                with torch.no_grad():
                    query_embed = self.kwargs['roberta_model'](**query, output_hidden_states=True, return_dict=True).pooler_output
                scores = cosine_similarity(self.kwargs['train_embed'].cpu().numpy(), query_embed.cpu().numpy())
                scores_ = scores.squeeze(-1)
                # args.logger.info(scores)

                i = self.kwargs['ICL_examples_num']
                index_list = np.argsort(scores_).tolist()  # similarity index(from low to high)
                if self.kwargs and 'train_flag' in self.kwargs.keys(): # train, need to exclude item itself
                    train_example_cnt = 0
                    tmp_index = -1  # from right to left  -1 -2 -3
                    prefix_items = []
                    while train_example_cnt < i:
                        train_index = index_list[tmp_index]
                        # item = self.kwargs['train_ds'][train_index]  # 源代码是这个
                        item = self.data[train_index]  # 修改代码 

                        tmp_index -= 1
                        if item['mention'] == mention and item['text'] == text:  # the same as train item
                            continue
                        else:  # different with train item
                            prefix_items.append(item)  # similarity decreases from left to right
                            train_example_cnt += 1
                    prefix_items.reverse()  # similarity rises from left to right
                else:  # dev or test
                    sorted_index = index_list[-i:]  # from low to high, select ICL_examples_num items
                    # prefix_items = [self.kwargs['train_ds'][index] for index in sorted_index]  # 原始代码
                    prefix_items = [self.data[index] for index in sorted_index]  # 修改代码
                
                tmpDict['examples'] = prefix_items
                prefix_items_list.append(prefix_items)

            json.dump(prefix_items_list, open(example_file_path, 'w'), indent=2)


    def collate_fn(self, items):
        batch_pairs, batch_targets, batch_golden = [], [], []
        for item in items:
            batch_pairs.append(self._get_pairs(item))
            batch_targets.append(item['target'])
            batch_golden.append(item['golden'])
            return batch_pairs, batch_targets, batch_golden

    def _get_pairs(self, item):
        text, mention, image = item['text'], item['mention'], item['image']
        if self.kwargs['ICL_examples_num'] != 0:
            prefix_list = self._similar_prefix(item)
        else:
            prefix_list = []
        text_ = f'[Text]{text}  [Question]What does {mention} mentioned in the text refer to?  [Answer]'
        pair = (image, text_)
        pair_list = prefix_list + [pair]
        return pair_list

    def _similar_prefix(self, item):
        prefix_items = item['examples']
        prefix_list = []
        h = h5py.File(self.kwargs['img_file'], 'r')
        for demo in prefix_items:
            text_ = f'[Text]{demo["text"]}  [Question]What does {demo["mention"]} mentioned in the text refer to?  [Answer]{demo["target"]}  '
            # image = demo['image']
            img_url = demo['img_name'] if 'img_name' in demo.keys() else demo["img_url"]
            image = h[img_url][()]  # add image attribution to self.data

            prefix_list.append((image, text_))
        return prefix_list



