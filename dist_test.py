from tqdm import tqdm
import json
import os
import torch
from torch.utils.data import Dataset,Dataloader
from datasets import load_dataset,load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from huggingface_hub import HfApi, Repository
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
import pandas as pd
import csv
from implictdatareader import load_pdtb,transform_train_conversation,transform_test_conversation,transform_test_conversation_fewshot
from prompt import *
from accelerate import Accelerator

os.environ['TOKENIZERS_PARALLELISM']='False'

class CustomDataset(Dataset):
    def __init__(self,hf_dataset):
        self.hf_dataset=hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self,idx):
        return self.hf_dataset[idx]

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def main():
    accelerator=Accelerator()
    parser = argparse.ArgumentParser(description='parser for training decoder-only models')
    parser.add_argument('--model_name', type=str, default='facebook/opt-13b', help='opt and bloomz series')
    parser.add_argument('--testype', type=str, default='zeroshot', help='zeroshot or fewshot')
    parser.add_argument('--max_output_length', type=int, default=64, help='number of new output tokens for generation')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size during training and generation')
    args = parser.parse_args()

    ##text generation config
    tokenizer = AutoTokenizer.from_pretrained("/scratch/user/hanyun_yin/huggingface_model/"+args.model_name,cache_dir="/scratch/user/hanyun_yin/huggingface_model/"+args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
    "/scratch/user/hanyun_yin/huggingface_model/"+args.model_name,
    return_dict=True,
    repetition_penalty=1.5,
    cache_dir="/scratch/user/hanyun_yin/huggingface_model/"+args.model_name
)
    model.resize_token_embeddings(len(tokenizer),pad_to_multiple_of=8)

    if args.testype=="zeroshot":
        test_dataset=load_dataset('csv',data_files='./ins/testset_zeroshot.csv')
    elif args.testype=="fewshot":
        test_dataset=load_dataset('csv',data_files='./ins/testset_fewshot.csv')


    custom_dataset=CustomDataset(dataset['train'])
    test_loader=Dataloader(custom_dataset,batch_size=args.batch_size,shuffle=True)

    model,test_loader= accelerator.prepare(
    model,test_loader
)

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader)):
            
            accelerator.print(f"On test set, f1={m_f1}%, EM={m_em}%,rouge1={m_rouge1}, rouge2={m_rouge2}, rougeL={m_rougeL},rougeLsum={m_rougeLsum}********************************")
            
            total_count += batch_size
    
    print(f"On test set, rouge1={100*m_rouge1}, rouge2={100*m_rouge2}, rougeL={100*m_rougeL}, rougeLsum={100*m_rougeLsum}")
    print(f"On test set, f1={m_f1}%, EM={m_em}%")
 


if __name__ == "__main__":
    main()