# -*- coding: utf-8 -*-

# This script contains all data transformation and reading

import random
from torch.utils.data import Dataset
import math

senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
senttag2opinion = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}

aspect_cate_list = ['location general',
                    'food prices',
                    'food quality',
                    'food general',
                    'ambience general',
                    'service general',
                    'restaurant prices',
                    'drinks prices',
                    'restaurant miscellaneous',
                    'drinks quality',
                    'drinks style_options',
                    'restaurant general',
                    'food style_options']


def read_line_examples_from_file(data_path, silence=True):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    if silence:
        print(f"Total examples = {len(sents)}")
    return sents, labels


def get_para_aste_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        all_tri_sentences = []
        for tri in label:
            # a is an aspect term
            if len(tri[0]) == 1:
                a = sents[i][tri[0][0]]
            else:
                start_idx, end_idx = tri[0][0], tri[0][-1]
                a = ' '.join(sents[i][start_idx:end_idx+1])

            # b is an opinion term
            if len(tri[1]) == 1:
                b = sents[i][tri[1][0]]
            else:
                start_idx, end_idx = tri[1][0], tri[1][-1]
                b = ' '.join(sents[i][start_idx:end_idx+1])

            # c is the sentiment polarity
            c = senttag2opinion[tri[2]]           # 'POS' -> 'good'

            one_tri = f"It is {c} because {a} is {b}"
            all_tri_sentences.append(one_tri)
        targets.append(' [SSEP] '.join(all_tri_sentences))
    return targets


def get_para_tasd_targets(sents, labels):

    targets = []
    for label in labels:
        all_tri_sentences = []
        for triplet in label:
            at, ac, sp = triplet

            man_ot = sentword2opinion[sp]   # 'positive' -> 'great'

            if at == 'NULL':
                at = 'it'
            one_tri = f"{ac} is {man_ot} because {at} is {man_ot}"
            all_tri_sentences.append(one_tri)

        target = ' [SSEP] '.join(all_tri_sentences)
        targets.append(target)
    return targets

def get_quad(quad, order):
    new_list = []
    order_list = order.split(" ")
    for each in order_list:
        new_list.append(quad[each])
    return new_list

def get_para_asqp_targets(sents, labels, order):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    new_sents = []
    data_count = {}
    for i in range(len(sents)):
        label = labels[i]
        cur_sent = sents[i]

        if len(label) in data_count:
            data_count[len(label)] += 1
        else:
            data_count[len(label)] = 1
        if len(label) == 1:
            all_quad_sentences = []
            for quad in label:
                at, ac, sp, ot = quad

                man_ot = sentword2opinion[sp]  # 'POS' -> 'good'

                if at == 'NULL':  # for implicit aspect term
                    at = 'it'

                quad = {"[AT]": f"{at}",
                             "[OT]": f"{ot}",
                             "[AC]": f"{ac}",
                             "[SP]": f"{man_ot}"}
                quad_list = get_quad(quad, order)
                one_quad_sentence = ", ".join(quad_list)
                all_quad_sentences.append(one_quad_sentence)

            target = ' [SSEP] '.join(all_quad_sentences)
            targets.append(target)
            new_sents.append(cur_sent)

        else:
            all_quad_sentences = []
            for quad in label:
                at, ac, sp, ot = quad

                man_ot = sentword2opinion[sp]  # 'POS' -> 'good'

                if at == 'NULL':  # for implicit aspect term
                    at = 'it'

                quad = {"[AT]": f"{at}",
                        "[OT]": f"{ot}",
                        "[AC]": f"{ac}",
                        "[SP]": f"{man_ot}"}
                quad_list = get_quad(quad, order)
                one_quad_sentence = ", ".join(quad_list)
                all_quad_sentences.append(one_quad_sentence)

            target = ' [SSEP] '.join(all_quad_sentences)
            targets.append(target)
            new_sents.append(cur_sent)

    #print(data_count)
    return new_sents, targets

def get_para_asqp_targets_test(sents, labels, order):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    for label in labels:
        all_quad_sentences = []
        for quad in label:
            at, ac, sp, ot = quad

            man_ot = sentword2opinion[sp]  # 'POS' -> 'good'

            if at == 'NULL':  # for implicit aspect term
                at = 'it'

            quad = {"[AT]": f"{at}",
                    "[OT]": f"{ot}",
                    "[AC]": f"{ac}",
                    "[SP]": f"{man_ot}"}
            quad_list = get_quad(quad, order)
            one_quad_sentence = ", ".join(quad_list)
            all_quad_sentences.append(one_quad_sentence)

        target = ' [SSEP] '.join(all_quad_sentences)
        targets.append(target)
    return targets

def get_transformed_io(data_path, data_dir, data_type, order):
    """
    The main function to transform input & target according to the task
    """
    sents, labels = read_line_examples_from_file(data_path)

    """
    if data_type == "train":
        length = len(sents)
        half = math.floor(length / 8)
        sents = sents[0: half]
        labels = labels[0: half]
    """

    # the input is just the raw sentence
    inputs = [s.copy() for s in sents]

    task = 'asqp'
    if task == 'aste':
        targets = get_para_aste_targets(inputs, labels)
    elif task == 'tasd':
        targets = get_para_tasd_targets(inputs, labels)
    elif task == 'asqp':
        if data_type == "test":
            targets = get_para_asqp_targets_test(inputs, labels, order)
            #print(len(inputs), len(targets))
            return inputs, targets
        else:
            new_inputs, targets = get_para_asqp_targets(inputs, labels, order)
            #print(len(inputs), len(new_inputs), len(targets))
    else:
        raise NotImplementedError

    return new_inputs, targets


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, order, max_len=128):
        # './data/rest16/train.txt'
        self.data_path = f'../data/{data_dir}/{data_type}.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.data_type = data_type
        self.order = order

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def _build_examples(self):

        inputs, targets = get_transformed_io(self.data_path, self.data_dir, self.data_type, self.order)

        for i in range(len(inputs)):
            # change input and target to two strings
            input = ' '.join(inputs[i])
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
