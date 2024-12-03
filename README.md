# This is a repo of RL for IDRR.

Since the use of the Llama model requires authorization and the model weight file is too large to be included in the GitHub repository, you will need to obtain access to the model through Hugging Face and download the weights locally to execute the code.

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

Supervised Learning: ins_train.py

To perform supervised learning:
	1.	Download the model from Hugging Face.
	2.	Update the paths for the model and output in this file.
	3.	Run the ins_train.py file.

Inference: test.py

To evaluate the model’s performance:
	1.	Download the model from Hugging Face.
	2.	Update the paths for the tokenizer and model in this file.
	3.	Run the test.py file.

Reinforcement Learning: rlvr.py

To perform reinforcement learning:
	1.	Download the model from Hugging Face.
	2.	Update the model path in this file.
	3.	Run the rlvr.py file.



