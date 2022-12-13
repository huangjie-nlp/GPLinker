import torch
from models.models import GlobalLinker
from torch.utils.data import DataLoader
from dataloader.dataloader import MyDataset, collate_fn
from tqdm import tqdm
import json
import numpy as np
from utils.bert_optimization import BertAdam
# from loss.loss import multilabel_categorical_crossentropy

class Framework(object):
    def __init__(self, config):
        self.config = config
        with open(self.config.schema_fn, "r", encoding="utf-8") as f:
            self.id2label = json.load(f)[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        def sparse_multilabel_categorical_crossentropy(y_true=None, y_pred=None, mask_zero=False):
            '''
            稀疏多标签交叉熵损失的torch实现
            '''
            shape = y_pred.shape
            y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
            y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))
            zeros = torch.zeros_like(y_pred[..., :1])
            y_pred = torch.cat([y_pred, zeros], dim=-1)
            if mask_zero:
                infs = zeros + 1e12
                y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)
            y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
            y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
            if mask_zero:
                y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
                y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
            pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
            all_loss = torch.logsumexp(y_pred, dim=-1)
            aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
            aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-10, 1)
            neg_loss = all_loss + torch.log(aux_loss)
            loss = torch.mean(torch.sum(pos_loss + neg_loss))
            return loss

        dataset = MyDataset(self.config, self.config.train_fn)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=self.config.batch_size,
                                collate_fn=collate_fn, pin_memory=True)
        dev_dataset = MyDataset(self.config, self.config.dev_fn)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=1,
                                    collate_fn=collate_fn, pin_memory=True)

        model = GlobalLinker(self.config).to(self.device)

        def set_optimizer(model, train_steps=None):
            param_optimizer = list(model.named_parameters())
            param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=0.1, t_total=train_steps)
            return optimizer

        # optimizer = set_optimizer(model, train_steps=(int(len(dataloader) / self.config.batch_size + 1) * self.config.epochs))

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        best_epoch = 0
        best_f1_score = 0
        global_step = 0
        global_loss = 0
        p, r = 0, 0
        for epoch in range(self.config.epochs):
            for data in tqdm(dataloader):
                rel_logtis, head_logits, tail_logits = model(data)
                optimizer.zero_grad()
                rel_loss = sparse_multilabel_categorical_crossentropy(data["entity_list"].to(self.device), rel_logtis, True)
                head_loss = sparse_multilabel_categorical_crossentropy(data["head_list"].to(self.device), head_logits, True)
                tail_loss = sparse_multilabel_categorical_crossentropy(data["tail_list"].to(self.device), tail_logits, True)

                loss = sum([rel_loss + head_loss +tail_loss]) / 3
                loss.backward()
                optimizer.step()
                global_loss += loss.item()
                global_step += 1
            print("epoch {} global_step: {} global_loss: {:5.4f}".format(epoch, global_step, global_loss))
            global_loss = 0
            if (epoch + 1) % 5==0:
                precision, recall, f1_score, predict = self.evaluate(model, dev_dataloader)
                if best_f1_score < f1_score:
                    best_f1_score = f1_score
                    p, r = precision, recall
                    best_epoch = epoch
                    json.dump(predict, open(self.config.dev_result, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
                    print(
                        "epoch {} precision: {:5.4f} recall: {:5.4f} f1_score: {:5.4f} best_epoch: {}".format(epoch, p,
                                                                                                              r,
                                                                                                              f1_score,
                                                                                                              best_epoch))
                    print("save model......")
                    torch.save(model.state_dict(), self.config.checkpoint)
        print("best_epoch: {} precision: {:5.4f} recall: {:5.4f} f1_score: {:5.4f}".format(best_epoch, p, r, best_f1_score))

    def evaluate(self, model, dataloader, threshold=-0.5):

        model.eval()
        predict_num, gold_num, correct_num = 0, 0, 0
        predict = []

        def to_tuple(data):
            tuple_data = []
            for i in data:
                tuple_data.append(tuple(i))
            return tuple(tuple_data)

        with torch.no_grad():
            for data in dataloader:
                text = data["text"][0]
                token = data["token"][0]
                logits = model(data)
                outputs = [o.cpu()[0] for o in logits]
                outputs[0][:, [0, -1]] -= np.inf
                outputs[0][:, :, [0, -1]] -= np.inf

                subjects, objects = [], []
                for l, h, t in zip(*np.where(outputs[0] > threshold)):
                    if l == 0:
                        subjects.append((h, t))
                    else:
                        objects.append((h, t))

                spoes = []
                for sh, st in subjects:
                    for oh, ot in objects:
                        sp = np.where(outputs[1][:, sh, oh] >threshold)[0]
                        op = np.where(outputs[2][:, st, ot] > threshold)[0]
                        rs = set(sp) & set(op)
                        for r in rs:
                            sub = "".join(token[sh: st + 1])
                            relation = self.id2label[str(r)]
                            obj = "".join(token[oh: ot + 1])
                            spoes.append((sub, relation, obj))
                triple = data["spo_list"][0]
                triple = set(to_tuple(triple))
                pred = set(spoes)
                correct_num += len(triple & pred)
                predict_num += len(pred)
                gold_num += len(triple)
                lack = triple - pred
                new = pred - triple
                predict.append({"text": text, "gold": list(triple), "predict": list(pred),
                                "lack": list(lack), "new": list(new)})
            print("correct_num:{} predict_num: {} gold_num: {}".format(correct_num, predict_num, gold_num))
            recall = correct_num / (gold_num + 1e-10)
            precision = correct_num / (predict_num + 1e-10)
            f1_score = 2 * recall * precision / (recall + precision + 1e-10)
            print("precision: {:5.4f} recall: {:5.4f} f1_score: {:5.4f}".format(precision, recall, f1_score))
        return precision, recall, f1_score, predict