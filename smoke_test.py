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
        data_files={
            "train": "zip://tam_train.json::tam.zip",
            "test": "zip://tam_test.json::tam.zip",
        },
        features=features,
    )

    # subset for smoke test
    train_subset = dataset["train"].select(range(100))
    eval_subset = dataset["test"].select(range(20))

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

    train_tokenized = train_subset.map(
        preprocess,
        batched=True,
        remove_columns=train_subset.column_names,
    )
    eval_tokenized = eval_subset.map(
        preprocess,
        batched=True,
        remove_columns=eval_subset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="./smoke-test-output",
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="no",
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=10,
        logging_steps=2,
        fp16=False,
        use_cpu=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    print("smoke test passed")


if __name__ == "__main__":
    main()
