import torch
from transformers import BertForSequenceClassification, BertTokenizer
import json
import os

# ---------------------------------------------------------
# 1. Safe detection of base directory 
#    (works in both Jupyter Notebook and terminal)
# ---------------------------------------------------------
try:
    BASE_DIR = os.path.dirname(__file__)   # script execution
except NameError:
    BASE_DIR = os.getcwd()                 # Jupyter Notebook

# Path to the model directory
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "model"))

# ---------------------------------------------------------
# 2. Load tokenizer and model (LOCAL ONLY)
# ---------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

model = BertForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True   # IMPORTANT: ensures no HF download attempt
)

model.eval()

# ---------------------------------------------------------
# 3. Load id2label mapping
# ---------------------------------------------------------
id2label_path = os.path.join(MODEL_DIR, "id2label.json")
with open(id2label_path, "r") as f:
    id2label = json.load(f)

# ---------------------------------------------------------
# 4. Load sample sentences from file
# ---------------------------------------------------------
sample_file = os.path.abspath(os.path.join(BASE_DIR, "..", "demo", "sample_sentences.txt"))

with open(sample_file, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f.readlines() if line.strip()]

print("\nRunning flood-impact predictions on sample_sentences.txt...\n")


# ---------------------------------------------------------
# 5. Tokenize sentences
# ---------------------------------------------------------
inputs = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# ---------------------------------------------------------
# 6. Run model inference
# ---------------------------------------------------------
with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.sigmoid(outputs.logits)

# ---------------------------------------------------------
# 7. Threshold + label mapping
# ---------------------------------------------------------
THRESHOLD = 0.5
predictions = (probabilities > THRESHOLD).int()

# ---------------------------------------------------------
# 8. Print predicted categories
# ---------------------------------------------------------
for idx, sentence in enumerate(sentences):
    predicted_labels = [
        id2label[str(j)]
        for j in range(predictions.shape[1])
        if predictions[idx][j] == 1
    ]
    print(f"Sentence: {sentence}")
    print("Predicted Impact Categories:", predicted_labels)
    print("-" * 70)
