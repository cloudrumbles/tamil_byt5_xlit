import json
import zipfile
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)


def load_tamil_dataset(data_dir: Path = Path("./data")):
    """Download and load Tamil transliteration dataset."""
    data_dir.mkdir(exist_ok=True)

    train_path = data_dir / "tam_train.json"
    test_path = data_dir / "tam_test.json"

    if not train_path.exists():
        print("downloading dataset...")
        zip_path = hf_hub_download(
            repo_id="ai4bharat/Aksharantar",
            filename="tam.zip",
            repo_type="dataset",
        )
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)

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

    return DatasetDict({
        "train": Dataset.from_list(load_json(train_path), features=features),
        "test": Dataset.from_list(load_json(test_path), features=features),
    })


def main():
    model_name = "google/byt5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    dataset = load_tamil_dataset()

    def preprocess(examples):
        inputs = examples["english word"]
        targets = examples["native word"]

        model_inputs = tokenizer(
            inputs,
            max_length=128,
            truncation=True,
            padding=False,
        )

        labels = tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="./byt5-aksharantar-tamil",
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        learning_rate=1e-4,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_steps=500,
        logging_steps=100,
        bf16=True,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )

    eval_split = "validation" if "validation" in tokenized else "test"

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized[eval_split],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("./byt5-aksharantar-tamil-final")


if __name__ == "__main__":
    main()
