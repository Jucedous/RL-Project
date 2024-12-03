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
from implictdatareader import load_pdtb,transform_train_conversation,transform_test_conversation

@dataclass
class PPOArgs:
    # Model arguments
    model_name_or_path: str = field(default="/scratch/user/hanyun_yin/huggingface_model/Llama-3.2-1B-Instruct")
    #reward_model_path: str = field(default="EleutherAI/pythia-160m")
    reward_model_path: str = field(default="/scratch/user/hanyun_yin/huggingface_model/flan-t5-small")
    
    # Training arguments
    learning_rate: float = field(default=2e-5)
    total_episodes: int = field(default=100)
    per_device_train_batch_size: int = field(default=16)
    gradient_accumulation_steps: int = field(default=1)
    
    # PPO specific arguments
    num_epochs: int = field(default=5)
    mini_batch_size: int = field(default=16)
    beta: float = field(default=0.05)  # KL coefficient
    vf_coef: float = field(default=0.1)  # Value function coefficient
    cliprange: float = field(default=0.2)
    cliprange_value: float = field(default=0.2)
    gamma: float = field(default=1.0)  # Discount factor
    lam: float = field(default=0.95)  # GAE lambda
    whiten_rewards: bool = field(default=False)
    local_rollout_batch_size: int = field(default=16)
    
    # Generation config
    response_length: int = field(default=48)
    temperature: float = field(default=0.7)
    #output_dir: str = field(default="ppo_output")

def prepare_dataset(dataset, tokenizer, max_length=512):
    
    def tokenize(examples):
        # Tokenize inputs
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
    print("dataset successfully!!!")
    
    return tokenized_dataset

class CustomPPOTrainer(PPOTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        
    def compute_rewards(self, sequences, scores):
        """Compute rewards with KL penalty"""
        #kl = self.compute_kl(sequences)
        #non_score_reward = -self.args.beta * kl
        #rewards = scores + non_score_reward
        
        if self.args.whiten_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        return rewards
        
    def compute_kl(self, sequences):
        """Compute KL divergence between policy and reference model"""
        with torch.no_grad():
            policy_logprobs = self.model(sequences).logits
            ref_logprobs = self.ref_model(sequences).logits
            
            kl = F.kl_div(
                F.log_softmax(policy_logprobs, dim=-1),
                F.softmax(ref_logprobs, dim=-1),
                reduction='none'
            ).mean(-1)
            
        return kl

def main():
    parser = HfArgumentParser(PPOArgs)
    args, = parser.parse_args_into_dataclasses()
    
    # Initialize models
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Initialize PPO config with matched hyperparameters
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
    #ref_model = create_reference_model(model)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        num_labels=1,
        torch_dtype=torch.bfloat16
    )
    
    training_dataset=load_pdtb(split="train")
    training_dataset=training_dataset.map(transform_train_conversation)

    #dev_dataset=load_pdtb(split="dev")
    #dev_dataset=dev_dataset.map(transform_train_conversation)
    #test_dataset=load_pdtb(split="test")
    #test_dataset=test_dataset.map(transform_test_conversation)
    
    # Prepare dataset
    #dataset = prepare_dataset(dataset, tokenizer)
    training_dataset= prepare_dataset(training_dataset, tokenizer)
    
    # Initialize trainer
    trainer = CustomPPOTrainer(
        args=args,
        config=ppo_config,
        model=model,
        #ref_model=ref_model,
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
                temperature=args.temperature
            )
            
            print("generate successfully!!!")
            
            # Get rewards from reward model
            with torch.no_grad():
                rewards = torch.tensor(reward_model(response_tensors).logits.squeeze(-1))
            print(f"Rewards:{rewards}\n")
            
            # Compute full rewards (including KL penalty)
            rewards = trainer.compute_rewards(response_tensors, rewards)
            
            # Run PPO optimization
            stats = trainer.step(batch["input_ids"], response_tensors, rewards)
            
            # Log stats
            print(f"Epoch {epoch}, Stats: {stats}")
        
        # Save checkpoint
        trainer.save_pretrained("./rl_output")

if __name__ == "__main__":
    main()