#!/usr/bin/env python

"""This module will be used to fine-tune the model."""

from transformers import AutoTokenizer
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
import numpy as np


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def fine_tune_model(context_df, model_name, model_hub_name):
    """This function will fine-tune the model based on the provided parameters.
    Args:
        context_df (dataframe): This is the dataframe that will be used to fine-tune the model.
        model_name (string): This is the name you want to assign to the trained model.
        model_hub_name (string): This is the name of the Hugging Face repo to push the model to.
    Returns:
        success (string): This is a string that will be returned if the model is fine-tuned successfully.
    """

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_datasets = context_df.map(tokenize_function, batched=True)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    metric = load_metric("accuracy")

    # dataset train and eval splits using 80/20 split
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(frac=0.8)
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(frac=0.2)

    training_args = TrainingArguments(
        output_dir=model_name,
        evaluation_strategy="epoch",
        push_to_hub=True,
        push_to_hub_model_id=model_hub_name,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # push the model to huggingface hub
    trainer.push_to_hub()

    # push the tokenizer to huggingface hub
    tokenizer.push_to_hub(repo_id=model_hub_name)

    success = "Model fine-tuned successfully."
    return success
