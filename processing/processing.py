import json

file = "../data/CM/CMeIE_dev.json"
data = []
with open(file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        line_data = json.loads(line)
        text = line_data["text"]
        spo_list = line_data["spo_list"]
        temp = []
        for i in spo_list:
            subject = i["subject"]
            r = i["predicate"]
            subject_type = i["subject_type"]
            objects = i["object"]["@value"]
            object_type = i["object_type"]["@value"]
            temp.append((subject, subject_type + "/" + r + "/" + object_type, objects))
        data.append({"text": text, "spo_list": temp})
json.dump(data, open("../dataset/CMeIE/dev_data.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)