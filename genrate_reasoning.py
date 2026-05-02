import json, time, requests, threading, re
from queue import Queue, Empty


GROQ_KEYS_FILE = "groqkeys.txt"
INPUT_FILE = "data/sample_train.json"
OUTPUT_FILE = "reasoning_output.json"

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_WORKERS = 10

COMPONENTS = ["key_terms", "syntactic", "semantic", "pragmatic", "paraphrase"]


MASTER_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert linguistic analyst. Analyze the given Hindi sentence and generate 5 types of reasoning.

Return ONLY one valid JSON object with exactly these keys:
"key_terms", "syntactic", "semantic", "pragmatic", "paraphrase"

Strict output rules for every value:
- Do NOT start with phrases like "The sentence", "This sentence means", or similar.
- Each value must be a single clean sentence or phrase.
- Do not add explanations outside JSON.
- Do not mention that you are analyzing.
- Avoid redundancy, filler, and obvious restatements.
- If a field is not applicable, return an empty string "".

{
  "key_terms": "Identify the most important content words (nouns, verbs, key entities, domain-specific terms). Focus only on meaningful lexical units essential for translation. Avoid full sentence repetition.",
  
  "syntactic": "Analyze the syntactic structure using linguistic terms: grammatical relations, subject-object-verb roles, modifiers, clause structure, and word order. Do not start with words like 'sentence' or 'phrase'.",
  
  "semantic": "Provide a precise one-sentence semantic interpretation. Capture the exact meaning, resolve ambiguity, and ensure the intended sense of words is clear.",
  
  "pragmatic": "Analyze contextual and pragmatic aspects in one sentence: tone, intent, formality, audience, domain, and any implied meaning or communicative purpose.",
  
  "paraphrase": "Generate a semantically equivalent paraphrase in one sentence, improving clarity and reducing ambiguity while preserving exact meaning."
}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Sentence: "{INPUT_SENTENCE}"
Return valid JSON only.
<|eot_id|>"""


def load_keys():
    with open(GROQ_KEYS_FILE) as f:
        return [k.strip() for k in f if k.strip()]

keys = load_keys()
key_index = 0
key_lock = threading.Lock()

def get_key():
    global key_index
    with key_lock:
        k = keys[key_index]
        key_index = (key_index + 1) % len(keys)
        return k


with open(INPUT_FILE, encoding="utf-8") as f:
    data = json.load(f)

queue = Queue()
for i, row in enumerate(data):
    queue.put((i, row["source"]))

results = {}
write_lock = threading.Lock()


def clean_text(t):
    t = re.sub(r'[\x00-\x1F]', '', t)
    return t.replace('"', '\\"').strip()


def call_api(prompt, key):
    url = "https://api.groq.com/openai/v1/chat/completions"

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.05,
        "max_tokens": 700
    }

    for _ in range(5):
        try:
            r = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=40
            )

            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]

            time.sleep(2)

        except:
            time.sleep(2)

    return None


def worker(tid):
    while not queue.empty():
        try:
            idx, sentence = queue.get_nowait()
        except Empty:
            break

        key = get_key()
        safe_sentence = clean_text(sentence)

        prompt = MASTER_PROMPT.replace("{INPUT_SENTENCE}", safe_sentence)

        output = call_api(prompt, key)

        parsed = {k: "" for k in COMPONENTS}

        if output:
            try:
                clean = output.strip()

                if clean.startswith("```json"):
                    clean = clean[7:-3].strip()
                elif clean.startswith("```"):
                    clean = clean[3:-3].strip()

                obj = json.loads(clean)

                for k in COMPONENTS:
                    parsed[k] = obj.get(k, "")

            except:
                pass

        record = {
            "id": idx,
            "source": sentence,
            **parsed
        }

        with write_lock:
            results[idx] = record
            print(f"[T{tid}] Done {idx}")

        queue.task_done()


threads = []

num_threads = min(MAX_WORKERS, len(keys))

print(f"Starting {num_threads} workers...\n")

for i in range(num_threads):
    t = threading.Thread(target=worker, args=(i,))
    t.start()
    threads.append(t)
    time.sleep(0.3)

for t in threads:
    t.join()

# ─── SAVE ─────────────────────────────────
final = [results[i] for i in sorted(results)]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)

print("\nSaved:", OUTPUT_FILE)