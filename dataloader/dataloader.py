import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertTokenizer
import numpy as np
import torch


def search(token, part):
    part_list = list(part)
    part_len = len(part_list)
    for k, v in enumerate(token):
        if token[k: part_len+k] == part_list:
            return k
    return -1

def padding(l, lenth, t):
    """
    :param l:{{(0, 0),(0 ,0),...},{...},{...}}
    :param lenth: num
    :return:
    """
    to_list = [list(i) for i in l]
    array = []
    for i in to_list:
        temp = i
        if len(i) < lenth:
            for j in range(lenth - len(i)):
                temp.append((0, 0))
        else:
            if len(i) > 1:
                if len(i) != lenth:
                    print(lenth)
                    print(len(i))
                    print(i)
                    print(t)
                # assert len(i) == lenth
        array.append(temp)
    return np.array(array)

class MyDataset(Dataset):
    def __init__(self, config, fn):
        self.config = config
        with open(self.config.schema_fn, "r", encoding="utf-8") as f:
            self.label2id = json.load(f)[0]

        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path, do_lower_case=False)
        self.data = json.load(open(fn, "r", encoding="utf-8"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        json_data = self.data[idx]
        text = json_data["text"]
        token = ['[CLS]'] + list(text) + ['[SEP]']
        token_len = len(token)

        token2id = self.tokenizer.convert_tokens_to_ids(token)
        mask = [1] * token_len

        attention_mask = np.array(mask)
        input_ids = np.array(token2id)

        entity_list = [set() for _ in range(2)]
        head_list = [set() for _ in range(self.config.num_rel)]
        tail_list = [set() for _ in range(self.config.num_rel)]

        for spo in json_data["spo_list"]:
            subject = spo[0]
            predicate = spo[1]
            obj = spo[2]
            s_h_id = search(token, subject)
            o_h_id = search(token, obj)
            if o_h_id != -1 and s_h_id != -1:
                subject_id = (s_h_id, s_h_id + len(subject) - 1)
                object_id = (o_h_id, o_h_id + len(obj) - 1)
                relid = self.label2id[predicate]
                entity_list[0].add(subject_id)
                entity_list[1].add(object_id)
                head_list[relid].add((subject_id[0], object_id[0]))
                tail_list[relid].add((subject_id[1], object_id[1]))

        for label in entity_list +head_list +tail_list:
            if not label:
                label.add((0, 0))
        entity_list = [list(i) for i in entity_list]
        en1 = len(entity_list[0])
        en2 = len(entity_list[1])
        if en1 < en2:
            for i in range(en2-en1):
                entity_list[0].append([0, 0])
        else:
            for i in range(en1-en2):
                entity_list[1].append([0, 0])
        entity_list = np.array(entity_list)
        entity_list_length = entity_list.shape[1]

        head_length = [len(i) for i in head_list]
        max_hl = max(head_length)
        tail_length = [len(i) for i in tail_list]
        max_tl = max(tail_length)
        head_list = padding(head_list, max_hl, text)
        tail_list = padding(tail_list, max_tl, text)

        # entity_list = np.array([list(i) for i in entity_list])

        return text, token, json_data["spo_list"], input_ids, attention_mask, token_len, entity_list_length, \
               entity_list, head_list, tail_list, max_hl, max_tl

def collate_fn(batch):
    text, token, spo_list, input_ids, attention_mask, token_len, length, entity_list, head_list, tail_list, head_len, tail_len = zip(*batch)

    cur_batch = len(batch)
    max_text_length = max(token_len)
    max_length = max(length)
    max_head_len = max(head_len)
    max_tail_len = max(tail_len)

    batch_input_ids = torch.LongTensor(cur_batch, max_text_length).zero_()
    batch_mask = torch.LongTensor(cur_batch, max_text_length).zero_()
    batch_entity_list = torch.LongTensor(cur_batch, 2, max_length, 2).zero_()
    batch_head_list = torch.LongTensor(cur_batch, 53, max_head_len, 2).zero_()
    batch_tail_list = torch.LongTensor(cur_batch, 53, max_tail_len, 2).zero_()

    for i in range(cur_batch):
        batch_input_ids[i, :token_len[i]].copy_(torch.from_numpy(input_ids[i]))
        batch_mask[i, :token_len[i]].copy_(torch.from_numpy(attention_mask[i]))
        batch_entity_list[i, :, :length[i], :].copy_(torch.from_numpy(entity_list[i]))
        batch_head_list[i, :, :head_len[i], :].copy_(torch.from_numpy(head_list[i]))
        batch_tail_list[i, :, :tail_len[i], :].copy_(torch.from_numpy(tail_list[i]))

    return {"text": text,
            "input_ids": batch_input_ids,
            "attention_mask": batch_mask,
            "entity_list": batch_entity_list,
            "head_list": batch_head_list,
            "tail_list": batch_tail_list,
            "spo_list": spo_list,
            "token": token}

if __name__ == '__main__':
    from config.config import Config
    from torch.utils.data import DataLoader
    config = Config()
    dataset = MyDataset(config, config.train_fn)
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    for data in dataloader:
        print("*"*50)
        # print(data["entity_list"].shape)
        # print(data["head_list"].shape)
        # print(data["tail_list"].shape)
