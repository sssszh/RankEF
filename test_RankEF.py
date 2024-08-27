
import os
import pickle as pkl

from models.ModelBasedCodeT5 import CriticModelBasedCodeT5
import json
from transformers import RobertaTokenizer


import torch
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5p-770m-py")
model = CriticModelBasedCodeT5.from_pretrained("")
model.to(device)
softmax_fn = torch.nn.Softmax(dim=-1)
prob_list = os.listdir("/data/apps/test")
prob_list = sorted(prob_list)

for fname in tqdm(prob_list, ncols=0, total=len(prob_list)):
    prob_id = int(fname)
   
    prob_path = os.path.join("data/apps/test", fname)
    
    question_path = os.path.join(prob_path, 'question.txt')
    with open(question_path, 'r') as f:
        data = f.read()
    save_dict = {}
    code_list = json.load(open(f"code_path/{prob_id}.json", 'r'))[f'{prob_id}']["codes"]
    for idx, code_str in tqdm(enumerate(code_list), ncols=0, total=len(code_list)):
        input_str = "<CLASSIFICATION>" + "\nQUESTION:\n" + data + "\nCODE:\n" + code_str + "\nResult:\n"
        input_ids = tokenizer.encode(input_str)
        input_ids = input_ids[:1024]
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device) 
        class_labels = torch.LongTensor([1]).unsqueeze(0).to(device)
        output = model(input_ids=input_ids, class_labels=class_labels, return_dict=False)
        prob = float(softmax_fn(output[1])[0][0].detach())
        save_dict[f"{idx}"] = prob
    sort_list = sorted(save_dict, key=save_dict.get, reverse=True)
    save_list = []
    with open(f"pkl_path/{prob_id}.pkl", 'rb') as f:
        data_ = pkl.load(f)
    for id in sort_list[:20]:
        id = int(id)
        save_list.append(data_[prob_id]['results'][id])
    save_pkl = {prob_id:{"results":save_list}}
    with open(f"new_pkl_save_path/{prob_id}.pkl", 'wb') as f:
        pkl.dump(save_pkl, f)
