"""Module for actions with the GPT-Medium model available on Hugging Face"""

from transformers import AutoTokenizer, AutoModelForCausalLM


def save_model():
    """This function downloads and saves the model"""

    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    model.save_pretrained("microsoft/DialoGPT-medium")

    success = "Model saved successfully."

    return success


# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
