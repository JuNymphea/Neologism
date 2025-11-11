import os
import json
import argparse
import swanlab
from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)


class NeologismDataset(Dataset):
    """
    jsonl: 
    {
      "question": "...",
      "normal_answer": "...",   # rejected
      "concept_answer": "..."   # chosen
    }
    return:
      prompt_input_ids, prompt_attention_mask,
      chosen_input_ids, chosen_attention_mask,
      rejected_input_ids, rejected_attention_mask
    """

    def __init__(
        self,
        concept: str,
        tokenizer,
        new_token: str,
        max_prompt_length: int = 4096,
        max_completion_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.new_token = new_token
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length

        jsonl_path = f"neologism/data/train/{concept}.jsonl"

        self.data: List[Dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.data.append(
                    {
                        "question": obj["question"],
                        "normal_answer": obj["normal_answer"],
                        "concept_answer": obj["concept_answer"],
                    }
                )

    def __len__(self):
        return len(self.data)

    def _encode_completion(self, text: str):
        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_completion_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        return input_ids, attention_mask

    def __getitem__(self, idx):
        item = self.data[idx]
        q = item["question"]
        normal_answer = item["normal_answer"]
        concept_answer = item["concept_answer"]

        prompt_text = f"{q} Please {self.new_token} your answer."

        # prompt_enc = self.tokenizer(
        #     prompt_text,
        #     add_special_tokens=True,
        #     truncation=True,
        #     max_length=self.max_prompt_length,
        #     return_tensors="pt",
        # )
        # prompt_input_ids = prompt_enc["input_ids"][0]
        # prompt_attention_mask = prompt_enc["attention_mask"][0]

        # add chat_template
        prompt_enc = self.tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )
        prompt_input_ids = prompt_enc["input_ids"][0]
        prompt_attention_mask = prompt_enc["attention_mask"][0]

        chosen_input_ids, chosen_attention_mask = self._encode_completion(concept_answer)
        rejected_input_ids, rejected_attention_mask = self._encode_completion(normal_answer)

        return {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }

def collate_fn(batch, pad_token_id: int) -> Dict[str, torch.Tensor]:

    def pad_1d(key, pad_val):
        seqs = [b[key] for b in batch]
        return torch.nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=pad_val
        )

    prompt_input_ids = pad_1d("prompt_input_ids", pad_token_id)
    prompt_attention_mask = pad_1d("prompt_attention_mask", 0)

    chosen_input_ids = pad_1d("chosen_input_ids", pad_token_id)
    chosen_attention_mask = pad_1d("chosen_attention_mask", 0)

    rejected_input_ids = pad_1d("rejected_input_ids", pad_token_id)
    rejected_attention_mask = pad_1d("rejected_attention_mask", 0)

    return {
        "prompt_input_ids": prompt_input_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention_mask,
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": rejected_attention_mask,
    }

def _pad_to_length(t: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
    cur_len = t.size(1)
    if cur_len == target_len:
        return t
    pad_len = target_len - cur_len
    return F.pad(t, (0, pad_len), value=pad_value)


def _cat_prompts_and_completions(batch: dict, pad_token_id: int, device):
    prompt_input_ids = batch["prompt_input_ids"].to(device)           # [B, Lp]
    prompt_attention_mask = batch["prompt_attention_mask"].to(device)

    chosen_input_ids = batch["chosen_input_ids"].to(device)           # [B, Lc_chosen]
    chosen_attention_mask = batch["chosen_attention_mask"].to(device)

    rejected_input_ids = batch["rejected_input_ids"].to(device)       # [B, Lc_rej]
    rejected_attention_mask = batch["rejected_attention_mask"].to(device)

    B = prompt_input_ids.size(0)

    chosen_input_ids_full = torch.cat([prompt_input_ids, chosen_input_ids], dim=1)          # [B, Lp+Lc1]
    chosen_attention_mask_full = torch.cat([prompt_attention_mask, chosen_attention_mask], dim=1)

    rejected_input_ids_full = torch.cat([prompt_input_ids, rejected_input_ids], dim=1)      # [B, Lp+Lc2]
    rejected_attention_mask_full = torch.cat([prompt_attention_mask, rejected_attention_mask], dim=1)

    Tc = chosen_input_ids_full.size(1)
    Tr = rejected_input_ids_full.size(1)
    T = max(Tc, Tr)

    chosen_input_ids_full = _pad_to_length(chosen_input_ids_full, T, pad_token_id)
    rejected_input_ids_full = _pad_to_length(rejected_input_ids_full, T, pad_token_id)

    chosen_attention_mask_full = _pad_to_length(chosen_attention_mask_full, T, 0)
    rejected_attention_mask_full = _pad_to_length(rejected_attention_mask_full, T, 0)

    chosen_loss_mask = torch.cat(
        [torch.zeros_like(prompt_attention_mask), chosen_attention_mask],
        dim=1,
    )
    rejected_loss_mask = torch.cat(
        [torch.zeros_like(prompt_attention_mask), rejected_attention_mask],
        dim=1,
    )

    chosen_loss_mask = _pad_to_length(chosen_loss_mask.to(device), T, 0)
    rejected_loss_mask = _pad_to_length(rejected_loss_mask.to(device), T, 0)

    input_ids = torch.cat([chosen_input_ids_full, rejected_input_ids_full], dim=0)          # [2B, T]
    attention_mask = torch.cat([chosen_attention_mask_full, rejected_attention_mask_full], dim=0)
    loss_mask = torch.cat([chosen_loss_mask, rejected_loss_mask], dim=0).bool()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "num_examples": B,
    }

def concatenated_logps(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    cat = _cat_prompts_and_completions(batch, pad_token_id, device)

    input_ids = cat["input_ids"]            # [2B, T]
    attention_mask = cat["attention_mask"]  # [2B, T]
    loss_mask = cat["loss_mask"]            # [2B, T]
    num_examples = cat["num_examples"]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits                 # [2B, T, V]

    shift_logits = logits[:, :-1, :]             # [2B, T-1, V]
    shift_labels = input_ids[:, 1:]              # [2B, T-1]
    shift_loss_mask = loss_mask[:, 1:]           # [2B, T-1],

    log_probs = F.log_softmax(shift_logits, dim=-1)                     # [2B, T-1, V]
    per_token_logps = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)                                                       # [2B, T-1]

    per_token_logps = per_token_logps * shift_loss_mask.float()
    all_logps = per_token_logps.sum(dim=-1)                             # [2B],

    chosen_logps = all_logps[:num_examples]                             # [B]
    rejected_logps = all_logps[num_examples:]                           # [B]

    return {
        "chosen_logps": chosen_logps,
        "rejected_logps": rejected_logps,
    }

# =====================
#   APO-up Eq.(2) loss
# =====================

def apo_up_loss(
    policy_model: nn.Module,
    ref_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    pad_token_id: int,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Eq.(2):

    L(x, yc, yr) =
      - log σ( β log pθ(yc|x)/pθ(yr|x) + β log pθ0(yc|x)/pθ0(yr|x) )
      - log σ( β log pθ(yc|x)/pθ0(yc|x) )
    """

    device = next(policy_model.parameters()).device

    policy_out = concatenated_logps(policy_model, batch, pad_token_id)
    lp_theta_c = policy_out["chosen_logps"]     # log pθ(yc|x)
    lp_theta_r = policy_out["rejected_logps"]   # log pθ(yr|x)

    with torch.no_grad():
        ref_out = concatenated_logps(ref_model, batch, pad_token_id)
        lp_theta0_c = ref_out["chosen_logps"].to(device)   # log pθ0(yc|x)
        lp_theta0_r = ref_out["rejected_logps"].to(device) # log pθ0(yr|x)

    log_ratio_theta = lp_theta_c - lp_theta_r              # log pθ(yc)/pθ(yr)
    log_ratio_theta0 = lp_theta0_c - lp_theta0_r           # log pθ0(yc)/pθ0(yr)
    log_ratio_theta_theta0 = lp_theta_c - lp_theta0_c      # log pθ(yc)/pθ0(yc)

    term1_input = beta * (log_ratio_theta + log_ratio_theta0)
    term2_input = beta * log_ratio_theta_theta0

    loss1 = -F.logsigmoid(term1_input)
    loss2 = -F.logsigmoid(term2_input)

    loss = (loss1 + loss2).mean()
    return loss

class ApoUpTrainer(Trainer):
    def __init__(self, ref_model, pad_token_id, beta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.pad_token_id = pad_token_id
        self.beta = beta

        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        loss = apo_up_loss(
            policy_model=model,
            ref_model=self.ref_model,
            batch=inputs,
            pad_token_id=self.pad_token_id,
            beta=self.beta,
        )
        if return_outputs:
            return loss, {}
        return loss


def train(model_name, new_token, concept, output_dir, neutral_word, batch_size, num_epochs, lr, beta, seed):
    os.makedirs(output_dir, exist_ok=True)

    set_seed(seed)

    if torch.cuda.is_available() and torch.cuda.device_count() >= 2: 
        device_policy = torch.device("cuda:0") 
        device_ref = torch.device("cuda:1") 
    else: 
        device_policy = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        device_ref = device_policy

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    pad_token_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device_policy)
    model_ref = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device_ref)

    # add neologism to input_vocabulary
    if new_token in tokenizer.get_vocab():
        raise ValueError(f"Token {new_token} exists in the vocabulary!")
    tokenizer.add_tokens([new_token])

    new_id = tokenizer.convert_tokens_to_ids(new_token)

    assert new_id == len(tokenizer) - 1

    # initialize the embedding of new_token
    neutral_word = neutral_word
    neutral_ids = tokenizer(neutral_word, add_special_tokens=False)["input_ids"]
    neutral_id = neutral_ids[0]

    with torch.no_grad():
        for m in [model, model_ref]:
            m.config.tie_word_embeddings = False
            m.lm_head.weight = nn.Parameter(m.lm_head.weight.clone())
            emb_weight = m.get_input_embeddings().weight
            emb_weight[new_id] = emb_weight[neutral_id].clone()

    # freeze all the parameters of model
    for p in model.parameters():
        p.requires_grad = False

    emb = model.get_input_embeddings().weight
    emb.requires_grad = True

    train_ids = torch.tensor([new_id], device=emb.device)

    def grad_hook(grad):
        mask = torch.zeros_like(grad)
        mask[train_ids] = 1.0
        return grad * mask

    emb.register_hook(grad_hook)

    optimizer = AdamW([{"params": [emb], "lr": lr}])

    # freeze all the parameter of model_ref
    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad = False

    dataset = NeologismDataset(
        concept=concept,
        tokenizer=tokenizer,
        new_token=new_token,
    )

    def _collate(batch):
        return collate_fn(batch, pad_token_id=pad_token_id)

    swanlab.init(
        project="neologism-apo-up",
        experiment_name=f"{concept}_apo_up",
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="swanlab",
        bf16=torch.cuda.is_available(),
        seed=seed,
        data_seed=seed,
        gradient_checkpointing=True,
        gradient_accumulation_steps=8,
    )

    trainer = ApoUpTrainer(
        ref_model=model_ref,
        pad_token_id=pad_token_id,
        beta=beta,
        model=model,     
        args=training_args,
        train_dataset=dataset,
        data_collator=_collate,
        optimizers=(optimizer, None),
    )
    trainer.train()

    tokenizer.save_pretrained(f"{output_dir}/tokenizer")

    print(f"Training Finished. Model and tokenizer saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="neologism/model/google/gemma-3-4b-it"
    )
    parser.add_argument(
        "--concept",
        type=str,
        required=True
    )
    parser.add_argument(
        "--new_token",
        type=str,
        default="~juzhuoxuan"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="neologism/checkpoints"
    )
    parser.add_argument(
        "--neutral_word",
        type=str,
        default="accurate"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    output_dir = f"{args.output_dir}_{args.concept}_{args.neutral_word}_ct"

    train(args.model_name, args.new_token, args.concept, output_dir, args.neutral_word, args.batch_size, args.num_epochs, args.lr, args.beta, args.seed)

if __name__ == "__main__":
    main()
# TODO: add hinge loss + multiple template