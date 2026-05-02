import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_PATH = "./model/final"

TEST_FILE = "data/test_hindi.txt"
REF_FILE = "data/test_bengali.txt"

BATCH_SIZE = 16
OUT_PATH = "predictions.txt"

SELECTED_COMPONENTS = ["R3", "R5"]

COMPONENTS = {
    "R1": {"label": "Key Terms", "hi": "hindi_key_terms", "bn": "bengali_key_terms"},
    "R2": {"label": "Syntactic", "hi": "hindi_grammar", "bn": "bengali_grammar"},
    "R3": {"label": "Semantic", "hi": "hindi_main_message", "bn": "bengali_main_message"},
    "R4": {"label": "Pragmatic Context", "hi": "hindi_pragmatics", "bn": "bengali_pragmatics"},
    "R5": {"label": "Paraphrase", "hi": "hindi_paraphrase", "bn": "bengali_paraphrase"},
}

def load_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def predict_batch(prompts, tokenizer, model, max_new_tokens):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i, output in enumerate(outputs):
        generated = output[inputs["input_ids"].shape[1]:]
        pred = tokenizer.decode(generated, skip_special_tokens=True).strip().replace("\n", " ")
        results.append(pred)

    return results

def build_prompt(sentence):
    return (
        f"<start_of_turn>user\n"
        f"You are a professional Hindi-to-Bengali translator.\n"
        f"STRICT RULE: Output ONLY the Bengali translation. "
        f"No explanations, no English, no Hindi, no labels — Bengali script only.\n\n"
        f"[Source — Hindi]\n{sentence}\n\n"
        f"Now write the Bengali translation:"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}
)
model.eval()

sentences = load_lines(TEST_FILE)
ref_sentences = load_lines(REF_FILE)

ref_lengths = sorted([len(tokenizer.encode(s)) for s in ref_sentences])
p99 = ref_lengths[int(len(ref_lengths) * 0.99)]
MAX_NEW_TOKENS = p99 + 20

prompts = [build_prompt(s) for s in sentences]

print(f"Total sentences: {len(sentences)}")
print(f"Using max_new_tokens: {MAX_NEW_TOKENS}")

preview = predict_batch(prompts[:5], tokenizer, model, MAX_NEW_TOKENS)

for i, (src, pred) in enumerate(zip(sentences[:5], preview)):
    print(f"\n[{i+1}] Hindi: {src}")
    print(f"     Bengali: {pred}")

input("\nPress Enter to continue...")

all_preds = []

for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
    batch = prompts[i:i + BATCH_SIZE]
    all_preds.extend(predict_batch(batch, tokenizer, model, MAX_NEW_TOKENS))

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(all_preds) + "\n")

print(f"Saved predictions to {OUT_PATH}")