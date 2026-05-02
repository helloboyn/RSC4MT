import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

import json, torch, gc
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "google/gemma-3-1b-it"
INPUT_FILE = "train_reasoning_filtered.json"

PER_DEVICE_BATCH = 4
GRAD_ACCUM = 16

SELECTED_COMPONENTS = ["R3", "R5"]

COMPONENTS = {
    "R1": {"label": "Key Terms", "hi": "hindi_key_terms", "bn": "bengali_key_terms"},
    "R2": {"label": "Syntactic", "hi": "hindi_grammar", "bn": "bengali_grammar"},
    "R3": {"label": "Semantic", "hi": "hindi_main_message", "bn": "bengali_main_message"},
    "R4": {"label": "Pragmatic Context", "hi": "hindi_pragmatics", "bn": "bengali_pragmatics"},
    "R5": {"label": "Paraphrase", "hi": "hindi_paraphrase", "bn": "bengali_paraphrase"},
}

def load_data(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

def get_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    return model

def get_lora():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True,
    )

def get_sft_config(output_dir):
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=100,
        learning_rate=1e-4,
        bf16=True,
        logging_steps=50,
        save_steps=1000,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        max_length=650,
        packing=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        dataset_num_proc=4,
        dataset_kwargs={"skip_preparation": True},
    )

def format_prompt(row):
    reasoning_block = ""

    for comp in SELECTED_COMPONENTS:
        cfg = COMPONENTS[comp]
        reasoning_block += (
            f"[{cfg['label']}]\n"
            f"Hindi:   {row[cfg['hi']]}\n"
            f"Bengali: {row[cfg['bn']]}\n\n"
        )

    return (
        f"<start_of_turn>user\n"
        f"You are a professional Hindi-to-Bengali translator.\n"
        f"STRICT RULE: Output ONLY the Bengali translation. "
        f"No explanations, no English, no Hindi, no labels — Bengali script only.\n\n"
        f"[Source — Hindi]\n{row['source']}\n\n"
        f"{reasoning_block}"
        f"Use the reasoning given above to help with the translation.\n"
        f"Now write the Bengali translation of the Hindi source sentence and nothing else:"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n{row['target']}<end_of_turn>"
    )

def cleanup():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

data = load_data(INPUT_FILE)

texts = [format_prompt(row) for row in data]
train_ds = Dataset.from_list([{"text": t} for t in texts])

name = "_".join(SELECTED_COMPONENTS)
output_dir = f"./gemma_lora_{name}"

tokenizer = get_tokenizer()
model = get_model()

trainer = SFTTrainer(
    model=model,
    args=get_sft_config(output_dir),
    train_dataset=train_ds,
    processing_class=tokenizer,
    peft_config=get_lora(),
)

trainer.train()
trainer.save_model(f"{output_dir}/final")
tokenizer.save_pretrained(f"{output_dir}/final")

cleanup()