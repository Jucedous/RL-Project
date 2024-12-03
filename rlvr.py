from dataclasses import dataclass, field
from typing import Optional, List
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments
)
from datasets import Dataset
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import LengthSampler
from tqdm import tqdm
import pandas as pd
import csv
from tqdm import tqdm
from implictdatareader import load_pdtb, transform_train_conversation, transform_test_conversation
from prompt import *
import pandas as pd
import csv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PPOArgs:
    model_name_or_path: str = field(default="[ path for the model ]")
    learning_rate: float = field(default=2e-5)
    total_episodes: int = field(default=100)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    num_epochs: int = field(default=1)
    mini_batch_size: int = field(default=1)
    beta: float = field(default=0.05)
    vf_coef: float = field(default=0.1)
    cliprange: float = field(default=0.2)
    cliprange_value: float = field(default=0.2)
    gamma: float = field(default=1.0)
    lam: float = field(default=0.95)
    whiten_rewards: bool = field(default=False)
    local_rollout_batch_size: int = field(default=1)
    response_length: int = field(default=10)
    temperature: float = field(default=0.75)
    #kl_penalty: float = field(default=0.1)

# Define label mapping
LABEL_MAP = {
    "Temporal": 0,
    "Comparison": 1,
    "Contingency": 2,
    "Expansion": 3
}

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(LABEL_MAP)

def to_one_hot_multi(labels, num_classes=NUM_CLASSES):
    """Convert multiple labels to one-hot vector"""
    one_hot = torch.zeros(num_classes)
    if ',' in labels:
        label_list = labels.split(',')
    else:
        label_list = [labels]
        
    for label in label_list:
        label_idx = LABEL_MAP[label.strip()]
        one_hot[label_idx] = 1
    return one_hot

def get_labels_from_one_hot(one_hot_tensor):
    """Convert one-hot tensor back to list of labels"""
    indices = torch.where(one_hot_tensor == 1)[0]
    return [REVERSE_LABEL_MAP[idx.item()] for idx in indices]

'''
def compute_reward(generated_text, one_hot_label, all_possible_answers):
    """
    Compute reward based on the generated text and answer types
    Args:
        generated_text (str): The generated response
        one_hot_label (tensor): One-hot encoded label tensor (can have multiple 1s)
        all_possible_answers (list): List of all possible answer types
    Returns:
        float: The reward value
    """
    correct_answers = get_labels_from_one_hot(one_hot_label)
    print(f"correct answer: {correct_answers}")
    
    answers=generated_text.split("### Response:")[5:]
    generated_text=" ".join(answers).strip().lower()
    print(f"generated response: {generated_text}")
    
    #tmp=answers=answers.split('###')[0].strip()
    #answers=answers.split('\n')[-1].strip()
    #if len(answers)>12:
        #answers=tmp.split('\n')[-3].strip()
    #pred_answer.append(answers)
    

    # Check if the generated text contains any of the correct answers
    for answer in correct_answers:
        if answer.lower() in generated_text.lower():
            return 1.0
            
    # Check if the generated text contains any other valid answer types
    other_answers = [ans for ans in all_possible_answers if ans not in correct_answers]
    for answer in other_answers:
        if answer.lower() in generated_text.lower():
            return -1.0
            
    # If the answer is not any of the valid types
    return -10.0
'''
def compute_reward(generated_text, one_hot_label, all_possible_answers):
    correct_answers = get_labels_from_one_hot(one_hot_label)
    
    answers = generated_text.split("### Response:")[5:]
    generated_text = " ".join(answers).strip().lower()
    
    # Base reward
    reward = 0.0
    
    # Check for correct answers
    found_correct = False
    for answer in correct_answers:
        if answer.lower() in generated_text.lower():
            reward += 1.0
            found_correct = True
    
    # Penalize incorrect answers, but less severely
    other_answers = [ans for ans in all_possible_answers if ans not in correct_answers]
    for answer in other_answers:
        if answer.lower() in generated_text.lower():
            reward -= 0.5
    
    # Additional penalties for degenerate text
    #if len(generated_text) < 5:  # Too short
        #reward -= 0.5
    if len(set(generated_text)) < 5:  # Too repetitive
        reward -= 0.5
        
    return max(-1.0, min(1.0, reward))  # Clamp rewards between -1 and 1

class PDTBDataset(TorchDataset):
    def __init__(self, tokenized, answers):
        self.input_ids = tokenized["input_ids"]
        self.attention_mask = tokenized["attention_mask"]
        self.answers = answers
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx].clone().detach(),
            'attention_mask': self.attention_mask[idx].clone().detach(),
            'answer': self.answers[idx].clone().detach()
        }

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    answers = torch.stack([item['answer'] for item in batch])
    
    return {
        'input_ids': input_ids.squeeze(),
        'attention_mask': attention_mask.squeeze(),
        'answer': answers
    }

def prepare_dataset(dataset, tokenizer, max_length=700):
    # Get all texts and answers at once
    all_texts = dataset["text"]
    
    # Convert string labels to one-hot encoding
    all_answers_one_hot = torch.stack([to_one_hot_multi(ans) for ans in dataset["answer"]])
    
    # Tokenize all texts at once
    tokenized = tokenizer(
        all_texts,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors="pt"
    )
    
    # Create custom dataset
    return PDTBDataset(tokenized, all_answers_one_hot)

def create_dataloader(dataset, batch_size, shuffle=False):
    """Create a PyTorch DataLoader from the dataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )

class CustomPPOTrainer(PPOTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        # Create reference model
        self.ref_model = create_reference_model(self.model)
    '''
    def compute_rewards(self, scores, logprobs, ref_logprobs, masks):
        """Compute per token rewards including KL penalty."""
        # Expand rewards to match sequence length
        bs, sl = masks.size()
        rewards = scores.unsqueeze(-1).expand(bs, sl).clone()  # expand to match mask size
        
        if self.args.whiten_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
        # No KL penalty
        #non_score_reward = torch.zeros_like(rewards)
        #kls = torch.zeros_like(rewards)

        # Calculate KL divergence
        kls = logprobs - ref_logprobs
        
        # Apply KL penalty
        non_score_reward = -0.3 * kls
        
        # Combine rewards with KL penalty
        total_rewards = rewards + non_score_reward
        
        return total_rewards, non_score_reward, kls
    '''
    def compute_rewards(self, scores, logprobs, ref_logprobs, masks):
        """Compute per token rewards including KL penalty."""
        # Expand rewards to match sequence length
        bs, sl = masks.size()
        rewards = scores.unsqueeze(-1).expand(bs, sl).clone()
        
        if self.args.whiten_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
        # Calculate KL divergence properly
        # KL(p||q) = p * (log(p) - log(q)) = p * log(p/q)
        # In log space: logp - logq
        kls = (logprobs.exp() * (logprobs - ref_logprobs)).clamp(min=0)
        
        # Apply KL penalty
        non_score_reward = -0.3 * kls
        
        # Combine rewards with KL penalty
        total_rewards = rewards + non_score_reward
        
        return total_rewards, non_score_reward, kls

def main():
    parser = HfArgumentParser(PPOArgs)
    args, = parser.parse_args_into_dataclasses()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
        kl_penalty='kl',  # Added KL penalty to config
        vf_coef=args.vf_coef,
        seed=42,
        optimize_cuda_cache=True
    )
    
    # Initialize model
    #model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name_or_path).to(device)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        "[ path for the model ]",
        return_dict=True,
        repetition_penalty=1.5).to(device)
    
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0.0  # Set dropout probability to 0

    # Load and prepare dataset
    training_dataset = load_pdtb(split="train")
    training_dataset = training_dataset.map(transform_train_conversation)
    training_dataset = training_dataset.map(lambda x: {'text': prompts_fewshot_template1 + x['text']})
    #print(training_dataset['text'][:1])
    df_fewshot = pd.DataFrame(training_dataset)
    df_fewshot.to_csv('./ins/trainset_fewshot.csv', index=False, encoding='utf-8')
    
    # Get all possible answer types 
    all_possible_answers = list(LABEL_MAP.keys())
    print(f"Label mapping: {LABEL_MAP}")
    
    # Prepare dataset and create dataloader
    prepared_dataset = prepare_dataset(training_dataset, tokenizer)
    train_dataloader = create_dataloader(
        prepared_dataset,
        batch_size=args.per_device_train_batch_size
    )
    
    # Initialize trainer
    trainer = CustomPPOTrainer(
        args=args,
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=prepared_dataset,
    )
    
    #with open('rltrain_log.txt', 'a') as file:  # Open the file in append mode
        # Training loop
    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            try:
                # Move batch to device and ensure proper dimensions
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Ensure input_ids and attention_mask are 2D
                if batch["input_ids"].dim() == 1:
                    batch["input_ids"] = batch["input_ids"].unsqueeze(0)
                if batch["attention_mask"].dim() == 1:
                    batch["attention_mask"] = batch["attention_mask"].unsqueeze(0)
                
                # Generate responses for the entire batch at once
                generation_kwargs = {
                    "do_sample": True,
                    "temperature": args.temperature,
                    "max_new_tokens": 10,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "top_p": 0.9,
                    "no_repeat_ngram_size": 3
                }
                
                # Print tensor shapes for debugging
                print(f"\nInput shapes - input_ids: {batch['input_ids'].shape}, attention_mask: {batch['attention_mask'].shape}")
                
                query_tensors=[]
                for i in batch["input_ids"]:
                    query_tensors.append(i)

                response_tensors = trainer.generate(
                    #batch["input_ids"],
                    query_tensors,
                    **generation_kwargs
                )
                
                # Decode generated responses
                decoded_responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
                #print(f"\nGenerated responses:")
                for resp in decoded_responses:
                    print(resp)
                
                # Compute rewards
                rewards = []
                for response, one_hot_label in zip(decoded_responses, batch["answer"]):
                    reward = compute_reward(response, one_hot_label, all_possible_answers)
                    rewards.append(torch.tensor(reward, device=device))
                
                #rewards = torch.stack(rewards)
                print(f"Batch {batch_idx} rewards: {rewards}\n")
                logger.info(f"Batch {batch_idx} rewards: {rewards}\n")
                
                # Run PPO step
                #stats = trainer.step(batch["input_ids"], response_tensors, rewards)
                stats = trainer.step(query_tensors, response_tensors, rewards)
                for key, value in stats.items():
                    logger.info(f"{key}: {value}\n")
                #print(f"Epoch {epoch}, Batch {batch_idx}, Stats: {stats}")
                
                #file.write(f"Epoch: {epoch}, Batch: {batch_idx}, Rewards: {rewards}, Stats: {stats}\n")
                if batch_idx % 100 == 0:
                    trainer.save_pretrained(f"./rl_output/3checkpoint-{epoch}-{batch_idx}")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                print(f"Batch shapes - input_ids: {batch['input_ids'].shape}, "
                    f"attention_mask: {batch['attention_mask'].shape}, "
                    f"answer: {batch['answer'].shape}")
                continue
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            del batch, response_tensors, rewards
        
        trainer.save_pretrained(f"./rl_output/3epoch-{epoch}")

if __name__ == "__main__":
    main()
