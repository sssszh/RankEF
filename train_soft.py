
import io
import logging
import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pprint
import sys
import time
import json
import pickle as pkl
import pdb 
import numpy as np
import scipy
from tqdm import tqdm
from datetime import datetime

import transformers
import torch

from Datasets.CodeT5BaseDataset import CodeT5BaseDataset
from trainers.trainer_soft import Trainer
from models.ModelBasedCodeT5 import CriticModelBasedCodeT5

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

def run_training(args, train_data, val_data):

    model_path = args.model_path if args.model_path is not None else '{}'.format(args.model)
    print("Loading model from {}...".format(model_path))

    model = CriticModelBasedCodeT5.from_pretrained(model_path)

    soft_model = transformers.T5ForConditionalGeneration.from_pretrained(model_path)
    soft_model.to(device)
    
    print('Finished loading model {}'.format(args.model))

    start_iteration = 0
    train_data.start_iteration = start_iteration
    print(f"Starting main loop")

    training_args = transformers.TrainingArguments(
        label_names=['labels', 'class_labels'],
        output_dir=args.save_dir,
        overwrite_output_dir=True, 
        remove_unused_columns=False,
        do_train=True,
        do_eval=False,
        do_predict=False,
        save_strategy='steps',
        eval_steps=200, 

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_steps=200,
        lr_scheduler_type='constant_with_warmup',

        logging_dir=args.save_dir, 
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_steps=args.save_freq,
        save_total_limit=args.save_total_limit,

        dataloader_drop_last=True,
        dataloader_num_workers=8,

        local_rank=args.local_rank,

        deepspeed=args.deepspeed,
        fp16=args.fp16,
        
    )
    trainer = Trainer(
        alpha=0.5,
        model=model,
        model_soft=soft_model,
        args=training_args,
        train_dataset=train_data,
    )
    trainer.train()

    if args.local_rank == 0:
        model.save_pretrained(os.path.join(args.save_dir, "final_checkpoint"))

def get_dataset(args, mode="train"): 
    
    if mode == "train": 
        dataroot = args.train_path
        with open(args.train_path, 'r') as f:
            problems_1 = f.readlines()
    elif mode == "val":
        dataroot = args.val_path
        with open(args.val_path, 'r') as f:
            problems_1 = f.readlines()
    
    # problems_2 = problems_2[:100]
    # train in debugging mode with small data split 
    if args.db and mode == "train":
        problems_1 = problems_1[:640]
    elif args.db and mode == "val":
        problems_1 = problems_1[:640]
    
    train_data = CodeT5BaseDataset(
        dataroot=dataroot,
        problems=problems_1,
        model=args.model,
        max_tokens=150,
        max_src_tokens=1024,
    )

    return train_data

def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset 
    train_data = get_dataset(args, "train")
    val_data = get_dataset(args, "val")

    # Save args to file
    json.dump(argsdict, open(os.path.join(args.save_dir, "args.json"), 'w'))

    # Load and train model; save model checkpoints 
    run_training(args, train_data, val_data)


if __name__ == "__main__":
    from configs.train_codet5_three_soft_configs import *
    
    main(args)