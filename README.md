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

## Impact Categories
  1: Aviation Impact
  2: Regular Business Loss
  3: Water Contamination
  4: Building damage
  5: Debris Impact
  6: Drowning Event
  7: Electrocution
  8: Evacuation
  9: Tree Fall
  10: Lightning
  11: Mental Health Issue
  12: Oil Industry
  13: Power Outage
  14: Road Transportation Issue
  15: Shipping Industry Impact
  16: Wildlife Encounter

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

## Citation
If you use this repository, please cite the associated manuscript.
