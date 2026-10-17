# Table-to-Text Generation with Few-Shot Prompted Qwen3-4B (ToTTo)

A **data-to-text generation** project. A table with highlighted cells goes in, and a single fluent, factual sentence describing those cells comes out. A 4-bit quantized **Qwen3-4B** model is few-shot prompted on the [ToTTo](https://huggingface.co/datasets/GEM/totto) benchmark and scored with **BLEU** and **BLEURT**.

Built as Project 2 for the **CENG 467** course. The full assignment description is in [`CENG467_Project2.pdf`](CENG467_Project2.pdf).

## How It Works

1. **Data:** the first 300 examples of the ToTTo validation split (`GEM/totto`).
2. **Table linearization:** each ToTTo table is converted to a Markdown table, and highlighted cells are wrapped in `<highlight>…</highlight>` tags.
3. **Prompting:** a system prompt with instructions, two hand-written few-shot examples, the linearized table and auxiliary context (page title, section title, section text).
4. **Generation:** Qwen3-4B loaded in 4-bit NF4 (bitsandbytes) with greedy decoding, `max_new_tokens=50`. The first generated line is kept as the prediction.
5. **Evaluation:** corpus BLEU (`sacrebleu`) and BLEURT-20 (`lucadiliello/BLEURT-20`).

## Experiments & Results

| Experiment | Script | Prompting strategy | BLEU | BLEURT |
| --- | --- | --- | --- | --- |
| Q1: Baseline | `CENG467_Project2_Q1.py` | Full table + `<highlight>` tags, 2-shot | **31.24** | **0.5467** |
| Q2.g: Custom prompt | `project_qa/CENG467_Project2_Q2_g.py` | Adds an explicit "Target Data Points" list of the highlighted values before the table | 31.24 | 0.5337 |
| Q3: Row tagging | `CENG467_Project2_Q3.py` | Rows containing highlighted cells are also marked with `<important>` | 31.24 | 0.5341 |

**Takeaways**

- The plain baseline prompt gave the best semantic quality (BLEURT). Extra structure in the prompt made a 4B-parameter model slightly *less* accurate, likely because it added noise to a long context.
- Compared with the ToTTo leaderboard, BLEU is well below fine-tuned SOTA models (~57.5), but the BLEURT score is competitive with several supervised baselines, without any task-specific training.
- Typical failures are hallucinated details that are not in the table (e.g. an invented production company) and misread column semantics (a throw distance described as "points").

The written answers to the project questions, including success and failure analyses and the BLEU vs. BLEURT discussion, are in [`project_qa/`](project_qa).

## Project Structure

```
.
├── CENG467_Project2_Q1.py          # Baseline few-shot generation + BLEU/BLEURT
├── CENG467_Project2_Q3.py          # Row-level <important> tagging experiment
├── calculate_q1_metrics.py         # Re-computes metrics from Q1_outputs_raw.json (batched BLEURT)
├── project_qa/
│   ├── CENG467_Project2_Q2_g.py    # Custom-prompt ("Target Data Points") experiment
│   ├── q2_a.txt … q2_g.txt         # Answers to Q2 sub-questions
│   └── q3_b.txt, q3_c.txt          # Answers to Q3 sub-questions
├── CENG467_Project2.pdf            # Assignment specification
└── requirements.txt
```

## Getting Started

**Requirements:** an NVIDIA GPU with CUDA. 4-bit loading through `bitsandbytes` needs a GPU.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Baseline (Q1): writes Q1_outputs_raw.json and Q1_outputs.json
python CENG467_Project2_Q1.py

# If generation finished but metric computation ran out of memory:
python calculate_q1_metrics.py

# Other experiments
python project_qa/CENG467_Project2_Q2_g.py   # writes Q2_outputs_g.json
python CENG467_Project2_Q3.py                # writes Q3_outputs.json
```

The model, dataset and BLEURT checkpoint download from the Hugging Face Hub on first run. Each output JSON holds the overall `metrics` plus a per-example list of `candidate` / `target` sentences.

## Tech Stack

Python · PyTorch · Hugging Face Transformers & Datasets · bitsandbytes · Accelerate · sacreBLEU · BLEURT (PyTorch)
