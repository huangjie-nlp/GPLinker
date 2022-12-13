
class Config(object):
    def __init__(self):
        self.dataset = "CIE"
        self.num_rel = 53
        self.batch_size = 8
        self.hidden_size = 64
        self.learning_rate = 2e-5
        # self.bert_path = "./RoBERTa_zh_Large_PyTorch"
        self.bert_path = "bert-base-chinese"
        self.train_fn = "./dataset/{}/train_data.json".format(self.dataset)
        self.dev_fn = "./dataset/{}/dev_data.json".format(self.dataset)
        self.schema_fn = "./dataset/{}/schema.json".format(self.dataset)
        self.checkpoint = "checkpoint/globalpointerlinker_adamw.pt"
        self.dev_result = "dev_result/dev.json"
        self.epochs = 100
        self.RoPE = True
        self.bert_dim = 768

