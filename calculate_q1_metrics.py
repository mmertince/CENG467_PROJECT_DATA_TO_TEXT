import json
import torch
import sacrebleu
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

def run_metrics_pipeline():
    # 1. Load intermediate results
    with open("Q1_outputs_raw.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    
    candidates = [res["candidate"] for res in results]
    # ToTTo targets are strings, sacrebleu expects a list of lists for references [cite: 38-39]
    targets = [[res["target"]] for res in results]
    raw_targets = [res["target"] for res in results]

    print(f"Calculating metrics for {len(candidates)} examples...")

    # 2. Calculate BLEU Score [cite: 20, 32, 39]
    print(f"length of candidates: {len(candidates)}")
    print(f"length of targets: {len(targets)}")
    bleu = sacrebleu.corpus_bleu(candidates, targets)
    final_bleu = bleu.score
    print(f"BLEU Score: {final_bleu:.4f}")

    # 3. Calculate BLEURT Score [cite: 20, 31, 39]
    
    print("Loading BLEURT-20 model...")
    bleurt_model_name = 'lucadiliello/BLEURT-20' 
    tokenizer_bl = BleurtTokenizer.from_pretrained(bleurt_model_name)
    model_bl = BleurtForSequenceClassification.from_pretrained(bleurt_model_name)
    model_bl.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_bl.to(device)

    bleurt_scores = []
    with torch.no_grad():
        # Process in batches to save memory
        for i in range(0, len(candidates), 10):
            batch_candidates = candidates[i:i+10]
            batch_targets = raw_targets[i:i+10]
            
            inputs = tokenizer_bl(batch_candidates, batch_targets, 
                                 padding=True, truncation=True, return_tensors="pt").to(device)
            
            res = model_bl(**inputs).logits.flatten().tolist()
            bleurt_scores.extend(res)
    
    final_bleurt = sum(bleurt_scores) / len(bleurt_scores)
    print(f"BLEURT Score: {final_bleurt:.4f}")

    # 4. Save the final JSON required by the project [cite: 231]
    final_data = {
        "metrics": {
            "bleu": final_bleu,
            "bleurt": final_bleurt
        },
        "results": results
    }
    
    with open("Q1_outputs.json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    
    print("Final results saved to Q1_outputs.json")

if __name__ == "__main__":
    run_metrics_pipeline()