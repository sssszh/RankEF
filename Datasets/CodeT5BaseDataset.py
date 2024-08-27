
import torch
import io
import numpy as np
import gc
import os
import random
from tqdm import tqdm 
from collections import Counter
import json, pdb 
from reindent import run as run_reindent

from multiprocessing import Manager
import transformers


class CodeT5BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, problems, model, max_tokens, max_src_tokens):
        self.dataroot = dataroot
        self.problems = problems 

        self.model = model
        
        self.max_tokens = max_tokens
        self.max_src_tokens = max_src_tokens

        self.samples = []           
        self.initialize()
        print("===================================================================================")
        print("load tokenizer:", model)

        self.tokenizer = transformers.RobertaTokenizer.from_pretrained(model)
    
    def initialize(self):

        all_samples = []
        skipped_problems = []

        all_samples_dict = {}

        print(f"Loading {len(self.problems)} problems from {self.dataroot}.")

        for idx, line in tqdm(enumerate(self.problems), ncols=0, total=len(self.problems)):
            json_line = json.loads(line)
            prompt = json_line["prompt"]
            code = json_line["code"]
            error_type = json_line["error_type"]
            error_message = json_line["error_message"]
            error_line_code = json_line["error_line_code"]
            error_line_number = json_line["error_line_number"]

            code = reindent_code(code)
            sample = (prompt, code, error_type, error_message, error_line_code, error_line_number)
            all_samples.append(sample)

        print(f"Loaded {len(all_samples)} saamples from {self.dataroot}.")
        # print(f"Skipped {len(skipped_problems)} problems from {self.dataroot}.")
        self.samples = all_samples


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        samples = self.pack_samples(idx)
        inputs = self.sample_complete_task(samples)

        gc.collect()
        return inputs

    def pack_samples(self, idx):
        """
        Repeatedly pick question, answer pairs from self.dataroot until we hit max_tokens.
        This will not include the tokens for the QUESTION and ANSWER prompt, as well as the  
        self.question_prefix. These will be added later and the total input will be 
        truncated if necessary.

        Always include the sample at idx at the beginning.
        """
        curr_num_tokens = 0
        curr_samples = [] 
        
        sample_pool = self.samples

        curr_p, curr_c, curr_et, curr_em, curr_elc, curr_eln = sample_pool[idx]

        while curr_num_tokens < self.max_tokens:

            curr_p = curr_p[:150000]
            curr_c = curr_c[:150000]
            curr_et = curr_et[:150000]
            if curr_em is not None:
                curr_em = curr_em[:150000]
            if curr_elc is not None:
                curr_elc = curr_elc[:150000]

            curr_num_tokens += len(self.tokenizer.tokenize(curr_p))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_c))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_et))
            if curr_em is not None:
                curr_num_tokens += len(self.tokenizer.tokenize(curr_em))
            if curr_elc is not None:
                curr_num_tokens += len(self.tokenizer.tokenize(curr_elc))

            curr_samples.append((curr_p, curr_c, curr_et, curr_em, curr_elc, curr_eln))
            break

        return curr_samples


    def sample_complete_task(self, samples):

        gen_input_ids = []
        class_input_ids = []
        gen_label_ids = []
        class_label_ids = []
    

        for sample in samples:

            p_str, c_str, et_str, em_str, elc_str, eln_str = sample
            g_p_str = "<GENERATION>" + "\nQUESTION:\n" + p_str + "\nCODE:\n" + c_str + "\nEVAL_MESSAGE:\n"
            c_p_str = "<CLASSIFICATION>" + "\nQUESTION:\n" + p_str + "\nCODE:\n" + c_str + "\nResult:\n"
            # GPT-Neo
            # g_p_str = "<GENERATION>" + p_str + c_str + "\nEVAL_MESSAGE:\n"
            # c_p_str = "<CLASSIFICATION>" + p_str + c_str + "\nResult:\n"
            
            if et_str == "correct":
                class_label_ids = [0]
                a_str = "This code is correct." + "\nResult:\n" + "Correct(0)"
            elif et_str == "OutputMismatch":
                class_label_ids = [2]
                a_str = "This code is wrong.\n" + "Error_Message:\n" + em_str + "\nError_TYPE:\n" + et_str + "\nResult:\n" + "Intent Error(2)"
            else:
                class_label_ids = [1]
                a_str = f"There is an error in the code at line {eln_str}:" + elc_str + "\nError_Message:\n" + em_str + "\nError_TYPE:\n" + et_str + "\nResult:\n" + "Execution Error(1)"

            gen_input_token_ids = self.tokenizer.encode(g_p_str, verbose=False)
            cls_input_token_ids = self.tokenizer.encode(c_p_str, verbose=False)
            gen_label_token_ids = self.tokenizer.encode(a_str, verbose=False)
            
            gen_input_ids.extend(gen_input_token_ids)
            class_input_ids.extend(cls_input_token_ids)
            gen_label_ids.extend(gen_label_token_ids)

        input_ids_max_len = self.max_src_tokens
        if len(gen_input_ids) < input_ids_max_len: 
            new_gen_input_ids = [self.tokenizer.eos_token_id] * input_ids_max_len
            new_gen_input_ids[:len(gen_input_ids)] = gen_input_ids
            gen_input_ids = new_gen_input_ids 

        if len(class_input_ids) < input_ids_max_len: 
            new_class_input_ids = [self.tokenizer.eos_token_id] * input_ids_max_len
            new_class_input_ids[:len(class_input_ids)] = class_input_ids
            class_input_ids = new_class_input_ids 
        
        if len(gen_label_ids) < self.max_tokens:
            new_gen_label_ids = [-100] * self.max_tokens 
            new_gen_label_ids[:len(gen_label_ids)] = gen_label_ids
            gen_label_ids = new_gen_label_ids
            
        gen_input_ids = gen_input_ids[:input_ids_max_len]
        class_input_ids = class_input_ids[:input_ids_max_len]
        gen_label_ids = gen_label_ids[:self.max_tokens]
        
        
        out_sample = {
            "input_ids" : torch.LongTensor(gen_input_ids),
            "labels" :  torch.LongTensor(gen_label_ids),
            "class_input_ids" : torch.LongTensor(class_input_ids),
            "class_labels": torch.LongTensor(class_label_ids),
        }            
            
        return out_sample


def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr, 
        ret, 
        config = {
            "dry-run": False,
            "help": False,
            "to": 4,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 4,
            "all-tabs": False
        }
    )

    return ret.getvalue()

