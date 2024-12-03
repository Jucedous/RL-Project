import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
#from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from implictdatareader import load_pdtb,transform_train_conversation,transform_test_conversation


################################################################################
# you need to change
model_name = "[ path for Llama-3.2-1B-Instruct ]"
# Output directory where the model predictions and checkpoints will be stored
output_dir = "[ output path ]"
################################################################################






# Number of training epochs
num_train_epochs = 10

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = True

# Batch size per GPU for training
per_device_train_batch_size = 1
#per_device_train_batch_size=6 if use_flash_attention else 4

# Batch size per GPU for evaluation
#per_device_eval_batch_size = 4
per_device_eval_batch_size = per_device_train_batch_size


# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-5

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
# lr_scheduler_type = "cosine"
# Learning rate schedule (constant a bit better than cosine)

lr_scheduler_type = "constant"


# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}


# Load dataset (you can process it here)
# dataset = load_dataset(dataset_name, split="train")
training_dataset=load_pdtb(split="train")
training_dataset=training_dataset.map(transform_train_conversation)
#dev_dataset=load_pdtb(split="dev")
#dev_dataset=dev_dataset.map(transform_train_conversation)
test_dataset=load_pdtb(split="test")
test_dataset=test_dataset.map(transform_test_conversation)


# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    repetition_penalty=1.5
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 
# Fix weird overflow issue with fp16 training


# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    #save_steps=save_steps,
    save_strategy="epoch",
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=training_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# # Train model
trainer.train()

# # Save trained model
trainer.save_model(output_dir)



