import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import sacrebleu
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from transformers import BitsAndBytesConfig

#bits and bytes config for 4bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

#set model and tokenizer
model_name = "Qwen/Qwen3-4B" #qwen3-4b model 
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) #trust remote code for custom tokenizer
#download model with half precision and automatic device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",
    quantization_config=bnb_config,#4bit quantization config
    trust_remote_code=True #trust remote code for custom model architecture
)

#load dataset first 300 validation samples as homework requires             
dataset = load_dataset("GEM/totto", split="validation[:300]")

#function to format table to markdown with highlighted cells
def format_table_to_markdown(table, highlighted_cells):
    #converts the ToTTo JSON table structure into a formatted Markdown string.
    #identifies cells in 'highlighted_cells' and wraps them with <highlight> tags.

    #initialize empyty table for formatted table
    markdown_table = []
    
    #iterate through each row and column in the table
    for row_idx, row in enumerate(table):    
        formatted_row = []
        is_row_highlighted = False #flag to check if the row has any highlighted cells
        for col_idx, cell in enumerate(row):
            cell_value = str(cell['value'])

            #check if the cell is highlighted
            is_highlighted = any(h[0] == row_idx and h[1] == col_idx for h in highlighted_cells)

            if is_highlighted:
                is_row_highlighted = True
                #wrap highlighted cell with <highlight> tags
                cell_value = f"<highlight>{cell_value}</highlight>"
                formatted_row.append(cell_value)
            else:
                formatted_row.append(cell_value)

        #conwert the row to markdown format
        formatted_row_str = "| " + " | ".join(formatted_row) + " |"

        if is_row_highlighted :
            #wrap the entire row with <important> tags if any cell is highlighted
            formatted_row_str = f"<important>{formatted_row_str}"
        #join the formatted row with markdown table
        markdown_table.append(formatted_row_str)

    return "\n".join(markdown_table)    

#define system prompt for the model based on hw requirements
SYSTEM_PROMPT = """
Your task is to generate a fluent sentence that describes the highlighted information in the provided table.

**Instructions:**
1. Read the **Input Table** and the **Auxiliary Context** (Page Title, Section Title, Section Text).
2. Focus **only** on the rows marked with `<important>` tags. Specifically,focus on cells wrapped in `<highlight>` and `</highlight>` tags.
3. Write a coherent, factual sentence that incorporates the information fromthese important rows with the highlighted cells.
4. You may use the Page Title or Section Title to provide necessary context (e.g., to identify the subject), but do not hallucinate details or include information from non-highlighted cells.
5. ONLY give a single sentence answer after the prompt.

---
"""

#define few shot example for the model based on hw requirements
FEW_SHOT_EXAMPLES = """
**Input Table:**
| # | Run | Title | Chapters | Author | Director | Ibope Rating |
| --- | --- | --- | --- | --- | --- | --- |
| 59 | June 5, 2000— February 2, 2001 | Laços de Família | 209 | Manoel Carlos | Ricardo Waddington | 44.9 |
| 60 | February 5, 2001— September 28, 2001 | Porto dos Milagres | 203 | Aguinaldo Silva Ricardo Linhares | Marcos Paulo Simões| 44.6 |
| 61 | October 1, 2001— June 14, 2002 | O Clone | 221 | Glória Perez | Jayme Monjardim | 47.0 |
| 62 | June 17, 2002— February 14, 2003 | Esperança | 209 | Benedito Ruy Barbosa | Luiz Fernando | 37.7 |
| 63 | February 17, 2003— October 10, 2003 | Mulheres Apaixonadas | 203 | Manoel Carlos | Ricardo Waddington | 46.6 |
| 64 | October 13, 2003— June 25, 2004 | Celebridade | 221 | Gilberto Braga | Dennis Carvalho | 46.0 |
| 65 | June 28, 2004— March 11, 2005 | Senhora do Destino | 221 | Aguinaldo Silva | Wolf Maya | 50.4 |
| 66 | March 14, 2005— November 4, 2005 | América | 203 | Glória Perez | Jayme Monjardim Marcos Schechtman | 49.4 |
| 67 | November 7, 2005— July 7, 2006 | Belíssima | 209 | Sílvio de Abreu | Denise Saraceni | 48.5 |
| 68 | July 10, 2006— March 2, 2007 | Páginas da Vida | 203 | Manoel Carlos | Jayme Monjardim | 46.8 |
| 69 | March 5, 2007— September 28, 2007 | Paraíso Tropical | 179 | Gilberto Braga Ricardo Linhares | Dennis Carvalho | 42.8 |
| 70 | October 1, 2007— May 31, 2008 | Duas Caras | 210 | Aguinaldo Silva | Wolf Maya | 41.1 |
<important>| 71 | June 2, 2008— January 16, 2009 | <highlight>A Favorita</highlight> | 197 | João Emanuel Carneiro | Ricardo Waddington |39.5 |
| 72 | January 19, 2009— September 11, 2009 | Caminho das Índias | 203 | Glória Perez | Marcos Schechtman | 38.8 |
| 73 | September 14, 2009— May 14, 2010 | Viver a Vida | 209 | Manoel Carlos | Jayme Monjardim | 35.6 |

**Auxiliary Context:**
* Page Title: List of 8/9 PM telenovelas of Rede Globo
* Section Title: 2000s
* Section Text: –

**Target Sentence:**
A Favorita is the telenovela aired in the 9 pm timeslot.
---

**Input Table:**
| State (class) | Vacator | Reason for change | Successor | Date of successor's formal installation |
| --- | --- | --- | --- | --- |
<important>| New Jersey (1) | <highlight>William Warren Barbour (R)</highlight> | <highlight>Died November 22, 1943. Successor was appointeduntil an election.</highlight> | <highlight>Arthur Walsh (D)</highlight> | <highlight>November 26, 1943</highlight> |
| Indiana (3) | Frederick Van Nuys (D) | Died January 25, 1944. Successor was appointed until an election. | Samuel D. Jackson(D) | January 28, 1944 |
| Massachusetts (2) | Henry Cabot Lodge, Jr. (R) | Resigned February 3, 1944, to go on active duty in the US Army. Success wasappointed until a special election. | Sinclair Weeks (R) | February 8, 1944 |
| Oregon (2) | Charles L. McNary (R) | Died February 25, 1944. Successor was appointed and subsequently won special election |Guy Cordon (R) | March 4, 1944 |
| Washington (3) | Homer Bone (D) | Resigned November 13, 1944, to become Judge of the U.S. Court of Appeals for the 9th Circuit.| Warren Magnuson (D) | December 14, 1944 |
| Indiana (3) | Samuel D. Jackson (D) | Appointee was not a candidate to finish the term. Successor was elected. | William E.Jenner (R) | November 7, 1944 |
| South Carolina (3) | Ellison Durant Smith (D) | Died November 17, 1944. | Wilton E. Hall (D) | November 20, 1944 |
<important>| New Jersey (1) | <highlight>Arthur Walsh (D)</highlight> | Appointee was not a candidate to finish the term. Successor waselected. | <highlight>H. Alexander Smith (R)</highlight> | <highlight>December 7, 1944</highlight> |
| Massachusetts (2) | Sinclair Weeks (R) | Appointee was not a candidate to finish the term. Successor was elected. | LeverettSaltonstall (R) | January 4, 1945 |

**Auxiliary Context:**
* Page Title: 78th United States Congress
* Section Title: Senate
* Section Text: –

**Target Sentence:**
William Warren Barbour (R) served as Senate until November 22, 1943, then by Arthur Walsh (D) from November 26, 1943 and was
succeeded by H. Alexander Smith (R), from December 7, 1944.

---
"""

#combines all parts to generate the final prompt for the model
def generate_prompt(example):

    #extract auxiliary context
    page_title = example.get('table_page_title', '-') #returns empty string if key not found
    section_title = example.get('table_section_title', '-') #returns empty string if key not found
    section_text = example.get('table_section_text', '-') #returns empty string if key not found

    #format table with highlighted cells
    markdown_table = format_table_to_markdown(
        example['table'],
        example['highlighted_cells']
    )

    #construct the final prompt for current tast
    current_task = f"""
**Input Table:**
{markdown_table}

**Auxiliary Context:**
* Page Title: {page_title}
* Section Title: {section_title}
* Section Text: {section_text}

**Target Sentence:**
"""
    #combine system prompt, few shot example and current task
    full_prompt = SYSTEM_PROMPT + FEW_SHOT_EXAMPLES + current_task
    return full_prompt

#outputs storage 
results = []

#loop first 300 samples in the dataset
limit = len(dataset)

for i in range(limit):
    example = dataset[i]
    
    #generate prompt for current example
    prompt = generate_prompt(example)
    
    #tokenize the prompt with truncation to 4096 tokens
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    
    #generate output from the model
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50, #based on hw requirements
            do_sample=False, #deterministic output
            pad_token_id=tokenizer.eos_token_id #if padding needed, use eos token
        )
    
    #only decode the generated part
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    #extract only the generated sentence after the prompt(remove prompt part)
    generated_sentence = generated_text[len(prompt):].strip().split('\n')[0]
    
    #store candidate and ground truth
    results.append({
        "example_index": i,
        "candidate": generated_sentence,
        "target": example['target']
    })

    #progress logging every example
    print(f"Processed {i+1} examples.")#bits and bytes config for 4bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("Saving intermediate results before metrics...")
with open("Q3_outputs_raw.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

#free up model memory
import gc
del model 
torch.cuda.empty_cache()
gc.collect()
    
#calculate BLEU and BLEURT scores
def calculate_metrics(results):
    candidates = [res['candidate'] for res in results]
    targets = [[res['target']] for res in results]  #sacrebleu expects list of references for each candidate

    #BLEU score calculation using sacrebleu
    bleu = sacrebleu.corpus_bleu(candidates, targets)
    print(f"--- Evaluation Results ---")
    print(f"BLEU score: {bleu.score:.2f}")

    #BLEURT score calculation(requires pre-trained BLEURT model)
    #using recommended checkpoint: BLEURT-20
    bleurt_model_name = 'lucadiliello/BLEURT-20'
    bleurt_tokenizer = BleurtTokenizer.from_pretrained(bleurt_model_name) #load tokenizer
    bleurt_model = BleurtForSequenceClassification.from_pretrained(bleurt_model_name) #load model
    bleurt_model = bleurt_model.to("cuda" if torch.cuda.is_available() else "cpu") #device mapping
    bleurt_model.eval() #evaluation mode

    with torch.no_grad():
        inputs = bleurt_tokenizer(candidates, [res["target"] for res in results],
                                    return_tensors="pt", padding=True, truncation=True).to(bleurt_model.device)
        
        bleurt_outputs = bleurt_model(**inputs).logits.flatten().tolist() #get BLEURT scores

    average_bleurt = sum(bleurt_outputs) / len(bleurt_outputs) #get average BLEURT score
    print(f"BLEURT score: {average_bleurt:.2f}") #print BLEURT score

    return bleu.score, average_bleurt

#calculate and print metrics
bleu_score, bleurt_score = calculate_metrics(results)

print("Processing complete. Results saved to Q3_outputs.json.")
#final dictionary including scores
final_output_data = {
    "metrics": {
        "bleu": bleu_score,
        "bleurt": bleurt_score
    },
    "results": results
}

#write to Q3_outputs.json
with open("Q3_outputs.json", "w", encoding="utf-8") as f:
    json.dump(final_output_data, f, ensure_ascii=False, indent=4)