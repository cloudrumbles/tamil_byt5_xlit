from datasets import load_dataset, Features, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)


def main():
    model_name = "google/byt5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    features = Features({
        "unique_identifier": Value("string"),
        "native word": Value("string"),
        "english word": Value("string"),
        "source": Value("string"),
        "score": Value("float64"),
    })

    dataset = load_dataset(
        "ai4bharat/Aksharantar",
        data_files={"train": "tam.zip", "test": "tam.zip"},
        features=features,
    )

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
