# NLP-BERT-Flood-Impact-Extraction
A NLP and deep learning framework for extracting flood-impact categories from news articles using a fine-tuned BERT-base-uncased model.

## Overview
This repository accompanies the methodology described in our paper:
“Comprehensive Flood Impact Assessment using NLP and BERT Deep Learning for Improved Impact Prediction and Disaster Management.”

It provides a minimal, reproducible demonstration of the impact-extraction component of the pipeline, including:
- A fine-tuned multi-label BERT model  
- Inference demo (`demo_predict.py`)  
- Sample input/output files  
- Supplementary pseudocode and workflow diagrams  
- A `requirements.txt` file for full reproducibility  

## Model Details
- Base model: bert-base-uncased  
- Framework: PyTorch + HuggingFace Transformers  
- Max sequence length during training: 128 tokens  
- Multi-label classification with 16 impact categories  
- Threshold for prediction: 0.5  
- Output dimension: 17 logits merged into 16 categories  

| # | Impact Category              | Example Sentence |
|---|------------------------------|------------------|
| 1 | Aviation Impact              | “Several flights were delayed or canceled due to the severe storm.” |
| 2 | Regular Business Loss        | “Local shops reported heavy financial losses after the flooding.” |
| 3 | Water Contamination          | “Residents were advised not to drink tap water due to contamination.” |
| 4 | Building Damage              | “Multiple homes suffered structural damage as floodwaters entered.” |
| 5 | Debris Impact                | “Floating debris blocked drainage channels and worsened flooding.” |
| 6 | Drowning Event               | “Emergency crews responded to reports of a drowning in the flooded area.” |
| 7 | Electrocution                | “Officials warned of electrocution risks from submerged power lines.” |
| 8 | Evacuation                   | “Dozens of families were evacuated as water levels continued rising.” |
| 9 | Tree Fall                    | “Fallen trees blocked several streets after the storm.” |
|10 | Lightning                    | “A lightning strike caused damage to nearby electrical equipment.” |
|11 | Mental Health Issue          | “Residents reported stress and anxiety following the storm impacts.” |
|12 | Oil Industry                 | “Refinery operations were disrupted due to flooding.” |
|13 | Power Outage                 | “Thousands of households experienced power outages overnight.” |
|14 | Road Transportation Issue    | “Major roadways were closed as floodwaters made them impassable.” |
|15 | Shipping Industry Impact     | “Port delays increased as cargo ships were unable to operate.” |
|16 | Wildlife Encounter           | “Residents spotted snakes and alligators moving through flooded neighborhoods.” |


## Quick Start
Install dependencies:
pip install -r requirements.txt

Run the demo:
python demo/demo_predict.py

## Running on Your Own Flood Reports
Modify the sentences list in sample_sentences.txt or use:

from transformers import BertTokenizer, BertForSequenceClassification
import torch, json

tokenizer = BertTokenizer.from_pretrained("model", local_files_only=True)
model = BertForSequenceClassification.from_pretrained("model", local_files_only=True)

# Load category names
with open("model/id2label.json") as f:
    id2label = json.load(f)

# Run prediction
inputs = tokenizer("YOUR SENTENCE HERE", return_tensors="pt", truncation=True, padding=True)
probs = torch.sigmoid(model(**inputs).logits)[0]

# Print categories with probabilities
for i, p in enumerate(probs):
    print(id2label[str(i)], float(p))


## Model Performance Summary
Precision: 0.85  
Recall: 0.90  
F1 Score: 0.87  
Epochs: 7  
Batch Size: 32  
Learning Rate: 2e-5  

## Sample Output
See demo/sample_output.json for full output format.

Example:
{
  "sentence": "Heavy flooding caused major road closures in southeast Houston.",
  "predicted_categories": ["Road Transportation Issue"]
}

## Repository Structure
model/            → Fine-tuned BERT weights + tokenizer  
demo/             → Inference demo + sample inputs/outputs  
supplementary/    → Pseudocode, diagrams, auxiliary files  
requirements.txt  → Required environment  
README.md         → Overview and instructions  

## Notes
- This repository demonstrates the **inference stage** of the workflow.  
- Full preprocessing, annotation, and training steps are documented in the manuscript and summarized in the pseudocode provided in `supplementary/`.
- The model includes 17 output labels because Aviation appears twice in the original schema. Both entries map to the same category and behave identically. In all analyses, they were merged into a single Aviation category (resulting in 16 final impact categories).

## Citation
If you use this repository, please cite the associated manuscript.
