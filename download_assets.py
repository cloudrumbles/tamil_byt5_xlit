"""Pre-download model and dataset for Docker build."""
from datasets import load_dataset, Features, Value
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/byt5-small"

print("downloading tokenizer...")
AutoTokenizer.from_pretrained(MODEL_NAME)

print("downloading model...")
AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("downloading dataset...")
features = Features({
    "unique_identifier": Value("string"),
    "native word": Value("string"),
    "english word": Value("string"),
    "source": Value("string"),
    "score": Value("float64"),
})

ds = load_dataset(
    "ai4bharat/Aksharantar",
    data_files={"train": "tam.zip", "test": "tam.zip"},
    features=features,
)

print(f"train: {len(ds['train'])} examples")
print(f"test: {len(ds['test'])} examples")
print("done")
