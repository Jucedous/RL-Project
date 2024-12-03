from tqdm import tqdm
import json
import os
import torch
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

os.environ['TOKENIZERS_PARALLELISM']='False'


def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


import argparse

parser=argparse.ArgumentParser(description='Test Type')
parser.add_argument('--testype',type=str,default='zeroshot',help='test type')
parser.add_argument('--model_name',type=str,default='Llama-3.1-8B-Instruct',help='model name')
args=parser.parse_args()

##load original test dataset
test_dataset=load_pdtb(split="test")

##map prompt to dataset
if args.testype=="fewshot":
    test_dataset_fewshot=test_dataset.map(lambda x: {'text':prompts_fewshot_template1+x['text']})
    df_fewshot=pd.DataFrame(test_dataset_fewshot)
    df_fewshot.to_csv('./ins/testset_fewshot.csv',index=False,encoding='utf-8')
elif args.testype=="zeroshot":
    test_dataset_zeroshot=test_dataset.map(transform_test_conversation)
    df_zeroshot=pd.DataFrame(test_dataset_zeroshot)
    df_zeroshot.to_csv('./ins/testset_zeroshot.csv',index=False,encoding='utf-8')

##create iterable dataset
#iter_test_dataset_zeroshot=test_dataset_zeroshot.to_iterable_dataset()
#iter_test_dataset_fewshot=test_dataset_fewshot.to_iterable_dataset()

##save instruction dataset to csv files
#df_zeroshot=pd.DataFrame(test_dataset_zeroshot)
#df_zeroshot.to_csv('./ins/testset_zeroshot.csv',index=False,encoding='utf-8')
#df_fewshot=pd.DataFrame(test_dataset_fewshot)
#df_fewshot.to_csv('./ins/testset_fewshot.csv',index=False,encoding='utf-8')

##an alternative way to save dataset to csv files
'''
with open('./ins/'+'zero_shot'".csv", 'w', encoding='utf-8') as file:
    writer=csv.writer(file)
    for sample in iter_test_dataset_zeroshot:
        print(sample)
        writer.writerow([sample])

with open('./ins/'+'few_shot'".csv", 'w', encoding='utf-8') as file:
    writer=csv.writer(file)
    for sample in iter_test_dataset_fewshot:
        writer.writerow([sample])
'''

##load dataset
if args.testype=="zeroshot":
    test_dataset=load_dataset('csv',data_files='./ins/testset_zeroshot.csv')
elif args.testype=="fewshot":
    test_dataset=load_dataset('csv',data_files='./ins/testset_fewshot.csv')

##label to id
batch_size=1
category_mapping = {'Temporal': 0, 'Comparison': 1, 'Conti': 2, 'Expansion': 3}

label_true=[]
for data in test_dataset['train']:
    label=[0,0,0,0]
    for category in category_mapping:
        if category.lower() in data['answer'].lower():
            label[category_mapping[category]]=1    
    label_true.append(label)


##text generation config
tokenizer = AutoTokenizer.from_pretrained("[ path for the model ]")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#tokenizer = AutoTokenizer.from_pretrained("/scratch/user/hanyun_yin/pdtb/results")
#tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    "[ path for the model ]",
    return_dict=True,
    repetition_penalty=1.5
)
model.resize_token_embeddings(len(tokenizer),pad_to_multiple_of=8)
model.eval()

##text generation
output_list=[]
with torch.no_grad():
    for i in tqdm(range(0, len(test_dataset['train']))):
        batch_prompts = test_dataset['train']['text'][i]
        model_inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True).to('cuda')
        model = model.cuda()
        outputs = model.generate(
            **model_inputs,
            do_sample=True,
            temperature=0.75,
            top_p=0.9,
            max_new_tokens=10
        )
        out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_list += out_sentence

ensure_directory_exists('./output/'+args.testype)
##save output to csv file
with open('./output/'+args.testype+'/'+args.model_name+".csv", 'w', encoding='utf-8') as file:
    writer=csv.writer(file)
    for output in output_list:
        writer.writerow([output])

##extract answers from output
def extract_answer(out_list,is_fewshot=True):
    pred_answer=[]
    for i,out_sentence in enumerate(out_list):
        if is_fewshot:
            answers=out_sentence.split("### Response:")[5:]
            answers=" ".join(answers).strip().lower()
            #tmp=answers=answers.split('###')[0].strip()
            #answers=answers.split('\n')[-1].strip()
            #if len(answers)>12:
                #answers=tmp.split('\n')[-3].strip()
            pred_answer.append(answers)
        else:
            answers=out_sentence.split("### Response:")[1:]
            answers=" ".join(answers).strip().lower()
            pred_answer.append(answers)

    print("len(out_list)",len(out_list))
    return pred_answer

pred_answer=extract_answer(output_list)
ensure_directory_exists('./answer/'+args.testype)
with open('./answer/'+args.testype+'/'+args.model_name+".csv", 'w', encoding='utf-8') as file:
    writer=csv.writer(file)
    for i,ans in enumerate(pred_answer):
        writer.writerow([i,ans])


cnt=0
label_predict=[]
exception_ans=[]
index_exc=[]
for i,ans in enumerate(pred_answer):
    label=[0,0,0,0]
    for category in category_mapping:
        if category.lower() in ans.lower():
            label[category_mapping[category]]=1
    if all(x==0 for x in label):
        cnt+=1
        index_exc.append(i)
        exception_ans.append(ans)   
    label_predict.append(label)


ensure_directory_exists('./exception_answer/'+args.testype)
with open('./exception_answer/'+args.testype+'/'+args.model_name+".csv", 'w', encoding='utf-8') as file:
    writer=csv.writer(file)
    for i,ans in enumerate(exception_ans):
        writer.writerow([index_exc[i],ans])

##metrics
def eval_result(label_predict,test_type):
    f1_macro = f1_score(label_true, label_predict, average='macro')
    print("f1_macro"+"_"+test_type,f1_macro)
    f1_micro = f1_score(label_true, label_predict, average='micro')
    print("f1_micro"+"_"+test_type,f1_micro)
    f1_weighted = f1_score(label_true, label_predict, average='weighted')
    print("f1_weighted"+"_"+test_type,f1_weighted)
    f1_none = f1_score(label_true, label_predict, average=None)
    print("f1_none"+"_"+test_type,f1_none)
    acc=accuracy_score(label_true,label_predict)
    print("accuracy_score"+'_'+test_type,accuracy_score(label_true, label_predict))
    return f1_macro,f1_micro,f1_weighted,acc,f1_none

f1_macro,f1_micro,f1_weighted,acc,f1_none=eval_result(label_predict,args.testype)
ensure_directory_exists('./results/'+args.testype)
with open('./results/'+args.testype+'/'+args.model_name+".csv",'w',encoding='utf-8') as file:
    writer=csv.writer(file)
    writer.writerow(["test dataset count:",len(output_list)])
    writer.writerow(["exception answer count:",cnt])
    writer.writerow(["f1_macro:",f1_macro])
    writer.writerow(["f1_micro:",f1_micro])
    writer.writerow(["f1_weighted:",f1_weighted])
    writer.writerow(["accuracy:",acc])
    writer.writerow(["per class F1:",f1_none])




