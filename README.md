# This is a repo of RL for IDRR.

## Setup

For setting up the environment, we recommend using virtual env + pip. The Python environment required is 3.10.8 (version)

 Install the packages given by
```bash
pip install -r requirements.txt
```

## important files

### data folder: pdtb_v2
### data loading function file: implictdatareader.py
### prompt template file: prompt.py

### supervised learning file: ins_train.py
to run supervised learning, just download model from huggingface, change the path of the model and output and then run the file.

### inference file: test.py
to evaluate the performance of the model, just download model from huggingface, change the tokenizer+model path and then run this file.

### reinforcement learning train file: rlvr.py
to use rl training, just download model from huggingface, change the path of the model and then run the file.



