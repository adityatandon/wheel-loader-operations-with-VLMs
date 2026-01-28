# Wheel Loader Operations with VLMs
End-to-end pipeline for wheel-loader operations using Vision+Language ideas:
1. Collect construction-site video frames
2. Auto-generate VQA (visual question answering) pairs + zero-shot object detection + action planning
3. Fine-tune an LLM with LoRA on the generated instruction data
4. Run inference in an interactive Q\&A loop with object detection context (optional)

> Note: Current code focuses on VQA-style instruction tuning and detection-based context. A “true VLM” that consumes pixels at inference time is not implemented end-to-end here; instead, images are used for object detection + dataset generation, and the LLM is fine-tuned on the resulting Q/A text.

---
## Base Language Model: TinyLLaMA
### Why TinyLLaMA?
This project uses **TinyLLaMA** as the default base model:
**Reasons for choosing TinyLLaMA:**
- Lightweight (~1.1B parameters)
- Fast fine-tuning with LoRA
- Can be fine-tuned on CPU

> Compute (or the lack of) is the MOST IMPORTANT consideration for using the Tiny version of the LLaMa model.
---

## Repository layout
- `scripts/wheel_dataloader.py` – download YouTube videos + extract quality frames
- `scripts/vqa_objectdetection.py` – generate VQA pairs (Moondream2) + detect objects (OWLv2) + action plan
- `scripts/llm_finetuning.py` – LoRA fine-tuning on JSONL VQA
- `scripts/inference_system.py` – inference CLI + interactive session

 Outputs:
- `data/frames/` – extracted frames
- `data/vqa/vqa\_dataset.jsonl` – training data (JSONL)
- `models/..../` – LoRA adapter weights + tokenizer

---

## Quickstart
### 1) Setup environment
```bash
python -m venv .venv # Create virtual environment
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run inference
```bash
python scripts/inference_system.py \
  --model models/tuned_model \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --interactive
```
