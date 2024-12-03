from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments
)
from datasets import Dataset, load_dataset
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import LengthSampler
from tqdm import tqdm
from implictdatareader import load_pdtb, transform_train_conversation, transform_test_conversation

@dataclass
class PPOArgs:
    # Previous arguments remain the same...
    model_name_or_path: str = field(default="/scratch/user/hanyun_yin/huggingface_model/Llama-3.2-1B-Instruct")
    #reward_model_path: str = field(default="/scratch/user/hanyun_yin/huggingface_model/Llama-3.2-1B-Instruct")
    learning_rate: float = field(default=2e-5)
    total_episodes: int = field(default=100)
    per_device_train_batch_size: int = field(default=16)
    gradient_accumulation_steps: int = field(default=1)
    num_epochs: int = field(default=5)
    mini_batch_size: int = field(default=16)
    beta: float = field(default=0.05)
    vf_coef: float = field(default=0.1)
    cliprange: float = field(default=0.2)
    cliprange_value: float = field(default=0.2)
    gamma: float = field(default=1.0)
    lam: float = field(default=0.95)
    whiten_rewards: bool = field(default=False)
    local_rollout_batch_size: int = field(default=16)
    response_length: int = field(default=48)
    temperature: float = field(default=0.7)

def prepare_dataset(dataset, tokenizer, max_length=512):
    def tokenize(examples):
        inputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return inputs
    
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names
    )
    print("Dataset prepared successfully!")
    return tokenized_dataset

class CustomPPOTrainer(PPOTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        
    def compute_rewards(self, sequences, scores):
        rewards = scores
        if self.args.whiten_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return rewards

def main():
    parser = HfArgumentParser(PPOArgs)
    args, = parser.parse_args_into_dataclasses()
    
    # Initialize models and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
        
    ppo_config = PPOConfig(
        model_name=args.model_name_or_path,
        learning_rate=args.learning_rate,
        batch_size=args.per_device_train_batch_size * args.gradient_accumulation_steps,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.num_epochs,
        gamma=args.gamma,
        lam=args.lam,
        cliprange=args.cliprange,
        cliprange_value=args.cliprange_value,
        vf_coef=args.vf_coef,
        seed=42,
        optimize_cuda_cache=True
    )
    
    # Initialize models
    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name_or_path)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        num_labels=1,
        torch_dtype=torch.bfloat16
    )
    
    # Load and prepare dataset
    training_dataset = load_pdtb(split="train")
    training_dataset = training_dataset.map(transform_train_conversation)
    training_dataset = prepare_dataset(training_dataset, tokenizer)
    
    # Initialize trainer
    trainer = CustomPPOTrainer(
        args=args,
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=training_dataset,
    )
    
    # Training loop
    for epoch in range(args.num_epochs):
        for batch in tqdm(trainer.dataloader):
            # Generate responses
            response_tensors = trainer.generate(
                batch["input_ids"],
                max_length=args.response_length,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            print("Generation completed successfully!")
            
            # Convert response tensors to appropriate format for reward model
            # Ensure responses are properly tokenized and padded
            responses_input = tokenizer(
                tokenizer.batch_decode(response_tensors, skip_special_tokens=True),
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=args.response_length
            )
            
            # Get rewards from reward model
            #with torch.no_grad():
                #reward_outputs = reward_model(**responses_input)
                #rewards = reward_outputs.logits.squeeze(-1)
            
            
            print(f"Rewards: {rewards}\n")
            
            # Compute full rewards
            rewards = trainer.compute_rewards(response_tensors, rewards)
            
            # Run PPO optimization
            stats = trainer.step(batch["input_ids"], response_tensors, rewards)
            
            print(f"Epoch {epoch}, Stats: {stats}")
        
        # Save checkpoint
        trainer.save_pretrained("./rl_output")

if __name__ == "__main__":
    main()
