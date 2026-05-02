import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json, time, threading, torch
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from groq import Groq

MODEL_PATH = "./model/final"
TEST_FILE = "data/test_hindi.txt"
REF_FILE = "data/test_bengali.txt"

OUT_PATH = "predictions_guided.txt"
CACHE_FILE = "reasoning_cache.json"

BATCH_SIZE = 16
MAX_WORKERS = 20

SELECTED_COMPONENTS = ["R3", "R5"]

COMPONENTS = {
    "R1": {"label": "Key Terms"},
    "R2": {"label": "Syntactic"},
    "R3": {"label": "Semantic"},
    "R4": {"label": "Pragmatic Context"},
    "R5": {"label": "Paraphrase"},
}

GROQ_KEYS_FILE = "groqkeys.txt"
MODEL_NAME_API = "meta-llama/llama-4-scout-17b-16e-instruct"

def load_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def load_keys(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

keys = load_keys(GROQ_KEYS_FILE)
clients = [Groq(api_key=k) for k in keys]
cycle_clients = cycle(clients)
lock = threading.Lock()

def get_client():
    with lock:
        return next(cycle_clients)

def get_reasoning(sentence, client):
    prompt = f"""Given this Hindi sentence, provide:
Semantic (Hindi): core meaning in one sentence
Paraphrase (Hindi): simpler rephrasing

Sentence: {sentence}

Format:
SEMANTIC: ...
PARAPHRASE: ..."""

    for _ in range(5):
        try:
            r = client.chat.completions.create(
                model=MODEL_NAME_API,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            text = r.choices[0].message.content.strip()
            lines = {
                l.split(":")[0].strip(): ":".join(l.split(":")[1:]).strip()
                for l in text.splitlines() if ":" in l
            }
            return lines.get("SEMANTIC", ""), lines.get("PARAPHRASE", "")
        except:
            time.sleep(2)
    return "", ""

def translate(text, client):
    if not text.strip():
        return ""
    for _ in range(5):
        try:
            r = client.chat.completions.create(
                model=MODEL_NAME_API,
                messages=[{
                    "role": "user",
                    "content": f"Translate to Bengali (only output Bengali):\n{text}"
                }],
                temperature=0.0
            )
            return r.choices[0].message.content.strip()
        except:
            time.sleep(2)
    return ""

def process(idx_sentence):
    idx, s = idx_sentence
    c = get_client()
    sem_hi, para_hi = get_reasoning(s, c)
    sem_bn = translate(sem_hi, c)
    para_bn = translate(para_hi, c)
    return idx, sem_hi, sem_bn, para_hi, para_bn

def build_prompt(s, sem_hi, sem_bn, para_hi, para_bn):
    block = ""

    if "R3" in SELECTED_COMPONENTS:
        block += f"[Semantic]\nHindi: {sem_hi}\nBengali: {sem_bn}\n\n"

    if "R5" in SELECTED_COMPONENTS:
        block += f"[Paraphrase]\nHindi: {para_hi}\nBengali: {para_bn}\n\n"

    return (
        f"<start_of_turn>user\n"
        f"You are a professional Hindi-to-Bengali translator.\n"
        f"STRICT RULE: Output ONLY Bengali.\n\n"
        f"[Source]\n{s}\n\n"
        f"{block}"
        f"Translate:\n"
        f"<end_of_turn>\n<start_of_turn>model\n"
    )

def predict_batch(prompts, tokenizer, model, max_new_tokens):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    res = []
    for i, out in enumerate(outputs):
        gen = out[inputs["input_ids"].shape[1]:]
        res.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    return res

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
ref = load_lines(REF_FILE)

lens = sorted([len(tokenizer.encode(s)) for s in ref])
MAX_NEW_TOKENS = lens[int(len(lens)*0.99)] + 20

if os.path.exists(CACHE_FILE):
    cache = json.load(open(CACHE_FILE))
else:
    cache = {}

reasonings = [None]*len(sentences)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = {}
    for i, s in enumerate(sentences):
        if s in cache:
            c = cache[s]
            reasonings[i] = (c["sem_hi"], c["sem_bn"], c["para_hi"], c["para_bn"])
        else:
            futures[ex.submit(process, (i, s))] = i

    for f in tqdm(as_completed(futures), total=len(futures)):
        i, sem_hi, sem_bn, para_hi, para_bn = f.result()
        s = sentences[i]
        reasonings[i] = (sem_hi, sem_bn, para_hi, para_bn)
        cache[s] = {
            "sem_hi": sem_hi,
            "sem_bn": sem_bn,
            "para_hi": para_hi,
            "para_bn": para_bn
        }

json.dump(cache, open(CACHE_FILE, "w"), ensure_ascii=False, indent=2)

prompts = [
    build_prompt(s, *r)
    for s, r in zip(sentences, reasonings)
]

all_preds = []

for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
    batch = prompts[i:i+BATCH_SIZE]
    all_preds.extend(predict_batch(batch, tokenizer, model, MAX_NEW_TOKENS))

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(all_preds))