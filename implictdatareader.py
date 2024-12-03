import torch
import os
import json
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from datasets import Dataset
from datasets import load_dataset,load_from_disk

def ImplicitDataset(foldlist):
    alt_cnt=0

    labels = []
    category_mapping = {'Temporal': 0, 'Comparison': 1, 'Contingency': 2, 'Expansion': 3}
    data=[]
    data_dict=defaultdict(list)
    file_path = './dataset/pdtb_v2/data/pdtb/'
    for idx in foldlist:
        fold_path=os.path.join(file_path,idx)
        for fold_file in sorted(os.listdir(fold_path)):
            filename=os.path.join(fold_path,fold_file)
            with open(filename, 'r', encoding='latin-1') as file:
                content = file.read()
            
            samples = content.split('________________________________________________________')[1:-1]
            
            for sample in samples:
                if "____AltLex____" in sample:
                    alt_cnt+=1
                # if "____Implicit____" not in sample and "____AltLex____" not in sample:
                if "____Implicit____" not in sample:
                    continue  # Skip non-implicit samples
                
                parts=sample.split('____Arg1____')
                #label = [0, 0, 0, 0]
                label=[]
                sup1=""
                arg1=""
                arg1_attr=""
                arg2=""
                arg2_attr=""
                sup2=""
                for category in category_mapping:
                    if category in parts[0]:
                        label.append(category)
                #         label[category_mapping[category]]=1
                        
                if "____Sup1____" in parts[0]:
                    sub_parts=parts[0].split('#### Text ####')
                    sup1=sub_parts[0].split('##############')[0].strip()
                arg_part=parts[1].split('____Arg2____')
                arg1_part=arg_part[0].split('#### Text ####')
                arg2_part=arg_part[1].split('#### Text ####')
                
                arg1 = "<ARG1>"+arg1_part[1].split('##############')[0].strip()+"</ARG1>"
                if len(arg1_part)>=3:
                    arg1_attr = "<ARG1_ATTR>"+arg1_part[2].split('##############')[0].strip()+"</ARG1_ATTR>"
                
                arg2 = "<ARG2>"+arg2_part[1].split('##############')[0].strip()+"</ARG2>"
                if len(arg2_part)>=3:
                    arg2_attr = "<ARG2_ATTR>"+arg2_part[2].split('##############')[0].strip()+"</ARG2_ATTR>"
                if len(arg2_part)>=4:
                    sup2=arg2_part[3].split('##############')[0].strip()
                answer=",".join(label)
                #data_dict.append({'text': f'<s>[INST] <<SYS>> Predict the relation between Arg1 and Arg2 enclosed in <ARG1></ARG1> and <ARG2></ARG2>. sup1 and sup2 is the supplement context. arg1_attr is the attribution of Arg1. arg2_attr is the is the attribution of Arg2<</SYS>>Given the document {sup1}{arg1}{arg1_attr}{arg2}{arg2_attr}{sup2} [/INST]</s>', 'answer':answer})
                #data_dict['text'].append(f'{sup1}{arg1}{arg1_attr}{arg2}{arg2_attr}{sup2}')
                ##p4
                data_dict['text'].append(f'{arg1}{arg2}')
                ##p5
                #data_dict['text'].append(f'{sup1}{arg1}{arg1_attr}{arg2}{arg2_attr}{sup2}')
                data_dict['answer'].append(answer)

                # data.append("sup1:{}. arg1:{}(atrribution of arg1: {}). arg2:{}(atrribution of arg2: {}). sup2:{}.".format(sup1,arg1,arg1_attr,arg2,arg2_attr,sup2))
                labels.append(answer)
    print("alt_cnt",alt_cnt)
    return data_dict

                
# Example usage
def load_pdtb(split="dev"):
    #training_fold_list = ['02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']
    training_fold_list = ['02']
    dev_fold_list = ['00','01']
    test_fold_list = ['21','22']
    if split=='train':
        datadict = ImplicitDataset(training_fold_list)
        dataset=Dataset.from_dict(datadict)
    if split=='dev':
        datadict = ImplicitDataset(dev_fold_list)
        dataset=Dataset.from_dict(datadict)
    if split=='test':
        datadict = ImplicitDataset(test_fold_list)
        dataset=Dataset.from_dict(datadict)
    return dataset

#Let's think step by step. Remember that at the final step, you need to output one of the four relations: Temporal, Comparison, Contingency, Expansion..
def transform_train_conversation(x):
    prompt= x['text']
    answer = x['answer']
    return {'text':f'''
### Instruction:
Given 2 sentences, your task is to classify the relation between them. Your output should be one of the four relations: Temporal, Comparison, Contingency, Expansion. 

### Input:
{prompt}

### Response:
{answer}

'''}
    
def transform_test_conversation_cot(x):
    prompt= x['text']
    answer = x['answer']
    ### cot with summary
    return {'text':f'''
### Instruction:
Given 2 sentences, your task is to classify the relation between them. Your output should be one of the four relations: Temporal, Comparison, Contingency, Expansion. Remember that you only need to output a word.

### Input:
{prompt}

### Response:
Let's think step by step, and give a summary at the end:
'''}


def transform_test_conversation(x):
    prompt= x['text']
    answer = x['answer']
    return {'text':f'''
### Instruction:
Given 2 sentences, your task is to classify the relation between them. Your output should be one of the four relations: Temporal, Comparison, Contingency, Expansion. 

### Input:
{prompt}

### Response:
'''}

def transform_test_conversation_fewshot(x):
    k_shot=2
    dataset=load_pdtb("train")
    prompt=""
    for i in range(k_shot):
        prompt+=transform_train_conversation(dataset[i])['text']
    return {'text':prompt+transform_test_conversation(x)['text']}
    
    