"""Pre-download model and dataset for Docker build."""
import json
import zipfile
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "google/byt5-small"
DATA_DIR = Path("/app/data")

print("downloading tokenizer...")
AutoTokenizer.from_pretrained(MODEL_NAME)

print("downloading model...")
AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("downloading dataset...")
DATA_DIR.mkdir(exist_ok=True)

zip_path = hf_hub_download(
    repo_id="ai4bharat/Aksharantar",
    filename="tam.zip",
    repo_type="dataset",
)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(DATA_DIR)

features = Features({
    "unique_identifier": Value("string"),
    "native word": Value("string"),
    "english word": Value("string"),
    "source": Value("string"),
    "score": Value("float64"),
})


def load_json(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


train_data = load_json(DATA_DIR / "tam_train.json")
test_data = load_json(DATA_DIR / "tam_test.json")

ds = DatasetDict({
    "train": Dataset.from_list(train_data, features=features),
    "test": Dataset.from_list(test_data, features=features),
})

print(f"train: {len(ds['train'])} examples")
print(f"test: {len(ds['test'])} examples")
print("done")
