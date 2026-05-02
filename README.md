# Reasoning as Supportive Context for Machine Translation  
### Hindi → Bengali Case Study

This repository contains the code and resources for an anonymous research submission.

---

## Overview

This work investigates whether structured reasoning can improve machine translation when used as **supportive context**.

We define five reasoning components:

- R1: Key Terms  
- R2: Syntactic  
- R3: Semantic  
- R4: Pragmatic  
- R5: Paraphrase  

A complete ablation over all **31 combinations** is conducted using a compact instruction-tuned model.

---

##  Key Findings

- Increasing reasoning components does not guarantee better performance  
- Performance depends on **type and composition**, not quantity  
- Best combination: **Semantic (R3) + Paraphrase (R5)**  
- Guided inference improves translation quality  

---

##  Core Idea

Reasoning is not generated as output but used as:

> **Supportive context to guide translation**

---

##  Dataset

- Training: 36,040 sentence pairs  
- Test: 2,000 sentence pairs  
- Multi-domain benchmark:
  - Agriculture
  - Tourism
  - Governance
  - Climate
  - Healthcare
  - Science & Technology
  - Judiciary
  - Education

---

##  Methodology

### Training
- Instruction-tuned base model  
- LoRA + Full fine-tuning  
- Exhaustive reasoning ablation  

### Inference Settings

1. Zero-Reasoning Inference  
2. Guided Inference (with optimal reasoning subset)  

---

##  Results Summary

- Large reasoning combinations degrade performance (**objective diffusion**)  
- Compact semantic signals perform best  
- Training–inference alignment is critical  

---

##  Tech Stack

- Python  
- PyTorch  
- HuggingFace Transformers  
- PEFT (LoRA)  
- TRL  

---

##  Installation

```bash
pip install -r requirements.txt