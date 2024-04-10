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
import collections
from copy import deepcopy

def compose_question(args, index, at, sentiment=None):
    sen = sentiment if sentiment is not None else "?"
    if args.pre_predict:
        if sentiment is None:  # query
            # at_text = f'[Question{index}]What is the sentiment of [text][{at}] and the answer format is "the sentiment of [text][{at}] is [emotion][{sen}]"'
            at_text = f'In the context, the [text][{at}] maybe the emotional words, please focus on it.'
        else: #incontext learning
            at_text = f'the sentiment of [text][{at}] is [emotion][{sen}], please focus on it.'
    else:
        if sentiment is None:  # query
            at_text = ""
        else: #incontext learning
            at_text = f'the sentiment of [text][{at}] is [emotion][{sen}]'


    return at_text

def compose_text(args, text, apsect_q):
    if args.pre_predict:
        text_ = f'\
            [Instruction]Given the context, Detect which words express emotions and corresponding emotion of the input utterance. The emotion is choosen from positive, negative and neutral.\
            [Clue]{apsect_q} \
            [Input]{text} \
            [Output]\
            '
    else:
        text_ = f'\
            [Instruction]Given the context, Detect which words express emotions and corresponding emotion of the input utterance. The emotion is choosen from positive, negative and neutral.\
            [Input]{text} \
            [Output]{apsect_q}\
            '
    return text_

class SADataset(Dataset):

    def __init__(self, file, tokenizer, args, aspect=None, **kwargs):
        self.file = file
        self.predict_aspect = aspect
        self.args = args
        self.tokenizer = tokenizer
        self.kwargs = kwargs

        self.data = self._get_data(file)

        refresh_img, refresh_exp = False, False
        if not os.path.exists(self.args.img_file) or refresh_img:   # Image file exist, handle image feature
            self._generate_image_feature()

        # if not os.path.exists(self.args.example_file) or refresh_exp:
        #     self._generate_example()
        self._generate_example()
        
        
        self._add_img_feat() # convert image url to clip output feature
        self.opinion = {0: 'negative', 1: 'neutral', 2: 'positive'}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def _get_data(self, file):
        num = 0
        file_name = file.split('/')[-1]
        sentence_d = collections.defaultdict(list)
        sentence_l, image_l, label_l, pair_l, senti_l, allabel_l = [], [], [] ,[] ,[], []
        with open(file, 'r', encoding="utf-8") as f:
            # Done split RT @ and etc
            while True:
                text = f.readline().rstrip('\n').split()
                # text = self.preprocess(text) # clean dataset, TODO re test
                if text == []:
                    break
                aspect = f.readline().rstrip('\n').split()
                sentiment = f.readline().rstrip('\n')
                image_path = f.readline().rstrip('\n')
                start_pos = text.index("$T$")
                end_pos = start_pos + len(aspect) - 1
                text = text[:start_pos] + aspect + text[start_pos+1:]
                sentence_d[" ".join(text)].append((start_pos, end_pos, sentiment, image_path))
            for key, value in sentence_d.items():

                text = key.split()
                sentence_l.append(text)
                n_key = len(text)
                s_label = [0] * n_key  # all is O
                s_senti = [-1] * n_key
                s_allabel = [0] * n_key
                s_pair = [] 
                image_l.append(value[0][3])
                for vv in value:
                    s_label[vv[0]] = 1  # B
                    for i in range(vv[0] + 1, vv[1] + 1):
                        s_label[i] = 2  # I

                    v_sentiment = int(vv[2]) + 1
                    for i in range(vv[0], vv[1]+1):
                        s_senti[i] = v_sentiment
                    # sentiment -1, 0, 1 -> 0, 1, 2

                    # 0代表实体外，1代表in，234代表实体开始及其情绪。 -1,0,1 ->2,3,4
                    s_allabel[vv[0]] = int(vv[2]) + 3 # B
                    for i in range(vv[0] + 1, vv[1] + 1):
                        s_allabel[i] = 1  # I

                    s_pair.append((str(vv[0]) + "-" + str(vv[1]), " ".join(text[vv[0]: vv[1]+1]), v_sentiment))  # [('4-5', 'Chuck Bass', 5), ('8-9', '# MCM', 4)]
                # print(text)  #['RT', '@', 'ltsChuckBass', ':', 'Chuck', 'Bass', 'is', 'everything', '#', 'MCM']
                # print(s_label)  # [0, 0, 0, 0, 1, 2, 0, 0, 1, 2]
                # print(s_senti)  # [-1, -1, -1, -1, 3, 3, -1, -1, 2, 2]
                # print(s_allabel)  # [0, 0, 0, 0, 4, 1, 0, 0, 3, 1]
                label_l.append(s_label)
                pair_l.append(s_pair)
                senti_l.append(s_senti)
                allabel_l.append(s_allabel)
        
        data = []
        # assert len(sentence_l) == len(self.aspect)

        for index, sent in enumerate(sentence_l):
            if len(pair_l[index]) >1:
                num += 1

            item = {
                "text": " ".join(sent),
                "sent": sent,
                "img_name": image_l[index],
                "label": label_l[index],
                "pair": pair_l[index],
                "senti": senti_l[index],
                "all_label": allabel_l[index],
                "predict_aspect": self.predict_aspect[index]
            }
            data.append(item)

        return data


    def _add_example(self):
        print("adding example into self.data")
        with open(self.args.example_file, 'r') as f:
            example_list = json.load(f)

        for tmpDict in self.data:
            tmpDict["examples"] = None
        for index, item in enumerate(self.data):
            item['examples'] = example_list[index]


    def _add_img_feat(self):
        # convert image url to clip output feature
        print("adding image feature into self.data")
        with h5py.File(self.args.img_file, 'r') as h:
            for tmpDict in self.data:
                if 'img_name' in tmpDict: # wikimel dataset
                    img_url = tmpDict['img_name']
                else: # wikidiverse dataset
                    img_url = tmpDict["img_url"]
                
                try:
                    tmpDict["image"] = h[img_url][()]  # add image attribution to self.data
                except:
                    print(f"{img_url} is missing")

    def collate_fn(self, items):
        batch_querys, batch_targets = [], []
        for item in items:
            batch_querys.append(self._get_query(item))
            batch_targets.append(self._get_targets(item['pair']))

        return batch_querys, batch_targets



    def _get_targets(self, target):
        all_text = []
        for t in target:
            index, aspect, sentiment = t
            text_ = f'the emotion of [text][{aspect}] is [emotion][{self.opinion[sentiment]}];'
            # text_ = f'([ASP][{aspect}], [SEN][{self.opinion[sentiment]}])'
            all_text.append(text_)
        all_text = " [SEP]".join(all_text)
        return all_text 

    

    def _get_query(self, item):
        text, image, example, aspect = item['text'], item['image'], item['examples'], item['predict_aspect']
        sent = item['sent']
        if self.args.ICL_examples_num != 0:
            prefix_list = self._incontext_prefix(item)
        else:
            prefix_list = []
        
        # print(text, "=====", aspect, "=====", item['pair'])
        aspect_question = []
        # print(sent)
        for index, a in enumerate(aspect):
            at = " ".join(sent[a[0]: a[1]+1])
            at_q = compose_question(self.args, index, at)
            aspect_question.append(at_q)
        aspect_question = " [SEP]".join(aspect_question)

        # text_ = f'Take a deep breath, Assess the sentiment in "{text}". The sentiment is choosen from positive, negative and neutral. [Question]{aspect_text}'
        # text_ = f'[Text]{text} [SEP] Assess the sentiment in text. The sentiment is choosen from positive, negative and neutral. [Question]{aspect_text}' #43.411
        text_ = compose_text(self.args, text, apsect_q=aspect_question)
        # print("query ", text_)
        pair = (image, text_)
        pair_list = prefix_list + [pair]

        return pair_list


    # In-context learning
    def _incontext_prefix(self, item):
        text, prefix_items = item['text'], item['examples']
        prefix_list, aspect_question = [], []
        h = h5py.File(self.kwargs['img_file'], 'r')

        for demo in prefix_items:
            for index, pair in enumerate(demo['pair']):
                _, aspect, sentiment = pair
                # at_text = f'the sentiment of [text][{aspect}] is [sentiment]{sentiment}'
                # at_text = f'the sentiment of [text][{aspect}] is [sentiment][{sentiment}]'
                at_q = compose_question(self.args, index, aspect, sentiment=self.opinion[sentiment])
                aspect_question.append(at_q)
            aspect_question = " [SEP]".join(aspect_question)

            # text_ = f'[Text]{text} [SEP]Please Assess the sentiment in text. The sentiment is choosen from positive, negative and neutral. [Question]{aspect_text} \n'
            # text_ = f'Please assess the sentiment in "{text}". The sentiment is choosen from positive, negative and neutral. [Question]{aspect_text}'
            text_ = compose_text(self.args, text, apsect_q=aspect_question)
            # print("incontext ", text_)
            img_url = demo['img_name'] if 'img_name' in demo.keys() else demo["img_url"]
            image = h[img_url][()]  # add image attribution to self.data

            prefix_list.append((image, text_))


        return prefix_list




    def _generate_example(self):
        data_cp = deepcopy(self.data)
        for item in tqdm(self.data, ncols=80, desc="generate ICL example"):
            item['examples'] = random.sample(data_cp, self.args.ICL_examples_num)




    def _generate_image_feature(self):
        print("Generating image feature")
        import clip
        import os
        from PIL import Image
        device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training
        Image.MAX_IMAGE_PIXELS = 2300000000  # 更改阈值像素上限

        img_list = os.listdir(self.args.path_image)
        image_embeddings = torch.FloatTensor(len(img_list), 512)
        model.eval()
        model.to(device)

        image_file_path = os.path.join(self.args.dataset_dir, "image_feature.h5")
        if os.path.exists(image_file_path):
            os.remove(image_file_path)
        image_file = h5py.File(os.path.join(self.args.dataset_dir, "image_feature.h5"), 'w')

        for index, item in tqdm(enumerate(img_list), total=len(img_list), desc="extracting image feature", ncols=100):
            img_path = os.path.join(self.args.path_image, item)

            try:
                image = Image.open(img_path)
                image = preprocess(image)
                image = image.unsqueeze(0).to(device)

                with torch.no_grad():
                    image_feature = model.encode_image(image).to(device)
                image_file[item] = image_feature.cpu().numpy()
            except Exception as e :
                print(e)
                print(item)
        json.dump(img_list, open(os.path.join(self.args.dataset_dir, "img_list.json"), 'w'), indent=2)
        image_file.close()
        # image_file = h5py.File(os.path.join(args.dataset_dir, "image_feature.h5"), 'w')
        # image_file.create_dataset("features", data=image_embeddings.numpy())


