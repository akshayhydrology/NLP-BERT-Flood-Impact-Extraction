# NLP-BERT-Flood-Impact-Extraction
A NLP and deep learning framework for extracting flood-impact categories from news articles using a fine-tuned BERT model.


## Overview
This repository accompanies the methodology described in our paper:
“Comprehensive Flood Impact Assessment using NLP and BERT Deep Learning for Improved Impact Prediction and Disaster Management.”

It provides a minimal, reproducible demonstration of the impact-extraction component of the pipeline.  
The repository includes the fine-tuned BERT model, an inference demo, and high-level pseudocode of the overall workflow.

## Key Components
- Fine-tuned BERT model for multi-label classification of 16 flood-impact categories.
- Inference demo (`demo_predict.py`) showing how to load the model and run predictions on sample sentences.
- Pseudocode summary** of the full workflow, including preprocessing, keyword filtering, annotation, training, and aggregation.
- Environment file (`requirements.txt`) to support reproducibility.


## Repository Structure
model/ → Fine-tuned BERT model and tokenizer files
demo/ → Minimal inference example
supplementary/ → Pipeline pseudocode and optional diagram
README.md → Project description
requirements.txt → Python dependencies


## Quick Start
To run the demo:

1. Install dependencies:
   `pip install -r requirements.txt`

2. Run the inference script:
   `python demo/demo_predict.py`

The script will load the trained model and print predicted flood-impact categories for the sample input sentences.

## Notes
- This repository demonstrates the **inference stage** of the workflow.  
- Full preprocessing, annotation, and training steps are documented in the manuscript and summarized in the pseudocode provided in `supplementary/`.
- The model includes 17 output labels because Aviation appears twice in the original schema. Both entries map to the same category and behave identically. In all analyses, they were merged into a single Aviation category (resulting in 16 final impact categories).

## Citation
If you use this repository, please cite the associated manuscript.
