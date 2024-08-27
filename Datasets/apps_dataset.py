
import torch
import glob
import logging
import random
import fnmatch
import numpy as np
import gc
import os
from tqdm import tqdm 
from collections import Counter
import pickle as pkl 
import json, pdb 

from multiprocessing import Manager
import transformers

import Datasets.utils as dsutils

class APPSBaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, problem_dirs, model, max_tokens, sample_mode,  max_src_tokens):
        self.dataroot = dataroot
        self.problem_dirs = problem_dirs 

        self.model = model
        self.sample_mode = sample_mode
        
        self.max_tokens = max_tokens
        self.max_src_tokens = max_src_tokens

        self.samples = []           
        self.initialize()

        if self.model in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py', 'codet5p-770m-py']:
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
            

    def load_gt_samples(self, sols, answer_type, starter_code, question_str):
        samples = []
        
        for sol_str in sols:
            sol_str = dsutils.reindent_code(sol_str)
            sample = (question_str, starter_code, sol_str, answer_type)
            samples.append(sample)
        
        return samples 

    def initialize(self):
        all_samples = []
        skipped_problems = []

        print(f"Loading {len(self.problem_dirs)} problems from {self.dataroot}.")
        for problem_name in tqdm(self.problem_dirs):
            question_fname = os.path.join(self.dataroot, problem_name, "question.txt")
            sols_fname = os.path.join(self.dataroot, problem_name, "solutions.json")            
            if (not os.path.isfile(question_fname)) or (not os.path.isfile(sols_fname)):
                skipped_problems.append(problem_name)
                continue
            
            with open(question_fname, 'r') as f:
                question_str = f.read()
            
            starter_code = os.path.join(self.dataroot, problem_name, "starter_code.py")    
            if (os.path.isfile(starter_code)):
                answer_type = "\nUse Call-Based format\n"
                with open(starter_code, 'r') as f:
                    starter_code = f.read()
            else:
                answer_type = "\nUse Standard Input format\n"
                starter_code = ""

            sols_str_list = json.load(open(sols_fname, 'r'))
            gt_samples = self.load_gt_samples(sols_str_list, answer_type, starter_code, question_str)
            all_samples += gt_samples

        print(f"Loaded {len(all_samples)} samples from {self.dataroot}.")
        print(f"Skipped {len(skipped_problems)} problems from {self.dataroot}.")

        self.samples = all_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_samples = self.pack_samples(idx)
        inputs = self.sample_task(raw_samples)

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
        
        if self.sample_mode == 'uniform_sol':
            curr_q, curr_s, curr_a, curr_q_prefix = sample_pool[idx]             
        elif self.sample_mode == 'uniform_prob':
            raise NotImplementedError()

        while curr_num_tokens < self.max_tokens:

            curr_q = curr_q[:150000]
            curr_s = curr_s[:150000]
            curr_a = curr_a[:150000]

            curr_num_tokens += len(self.tokenizer.tokenize(curr_q))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_s))            
            curr_num_tokens += len(self.tokenizer.tokenize(curr_a))
            
            curr_samples.append((curr_q, curr_s, curr_a, curr_q_prefix))
                
                # only pack 1 sample each sequence for codeT5 
            if self.model in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py', 'codet5p-770m-py']:
                break 

            if self.sample_mode == 'uniform_sol':
                new_idx = random.randint(0, len(sample_pool)-1)
                curr_q, curr_s, curr_a, curr_q_prefix = sample_pool[new_idx] 
            elif self.sample_mode == 'uniform_prob':
                raise NotImplementedError()

        return curr_samples

    def sample_task(self, samples, sample_type=None):

        input_ids = []
        label_ids = []
                    
        for sample in samples:
            q_str, s_str, a_str, answer_type = sample
            
            q_str =  "\nQUESTION:\n" + q_str + "\n" + s_str + "\n" + answer_type + "\nANSWER:\n"

            question_token_ids = self.tokenizer.encode(q_str, verbose=False)
            input_ids.extend(question_token_ids)
             
            answer_token_ids = self.tokenizer.encode(a_str, verbose=False)
            if self.model not in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py', 'codet5p-770m-py']:
                label_ids.extend([-100] * len(question_token_ids))
                answer_token_ids.append(self.tokenizer.eos_token_id)
                input_ids.extend(answer_token_ids)
            label_ids.extend(answer_token_ids)
                
        # Sanity checks and padding 
        input_ids_max_len = self.max_src_tokens if self.model in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py', 'codet5p-770m-py'] else self.max_tokens 
        if len(input_ids) < input_ids_max_len: 
            new_input_ids = [self.tokenizer.eos_token_id] * input_ids_max_len
            new_input_ids[:len(input_ids)] = input_ids
            input_ids = new_input_ids 
            
            if self.model not in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py', 'codet5p-770m-py']:
                new_label_ids = [-100] * input_ids_max_len 
                new_label_ids[:len(label_ids)] = label_ids
                label_ids = new_label_ids
                
        if self.model in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py', 'codet5p-770m-py'] and len(label_ids) < self.max_tokens:
            new_label_ids = [-100] * self.max_tokens 
            new_label_ids[:len(label_ids)] = label_ids
            label_ids = new_label_ids
        
        if self.model not in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py', 'codet5p-770m-py'] and len(input_ids) != len(label_ids): pdb.set_trace()
            
        # Cut off the excess
        input_ids = input_ids[:input_ids_max_len]
        label_ids = label_ids[:self.max_tokens]
        
        out_sample = {
            "input_ids" : torch.LongTensor(input_ids),
            "labels" :  torch.LongTensor(label_ids)
        }            
            
        return out_sample