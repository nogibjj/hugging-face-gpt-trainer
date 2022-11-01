"""The main control function for the model trainer."""

from logic.data_query import download_data, dataset_sampler
from logic.data_setup import available_characters, context_builder
from logic.gpt_model_finetune import fine_tune_model


def main():
    """Function to collect user input at desired intervals and return outputs"""

    # show a welcome message
    print(
        "\nWelcome to this Hugging Face GPT-Medium model trainer. \nIt will give you step-by-step prompts to fine-tune a model on a dataset you specify. \nA couple things to note: spelling counts, and you need to use Hugging Face to accomplish this. \nAt the end, we'll push the final model back to Hugging Face, so make sure you have a token set up."
    )

    # Ask user to define the location of the data on Hugging Face
    data_pathway = input(
        "\nPlease enter the location of the data on Hugging Face in this format: username/datasetname: "
    )

    # download the data
    df = download_data(data_pathway)

    # sample the data
    dataset_sample, dataset_cols = dataset_sampler(df)

    # show a sample of the dataset
    print(f"Here is a sample of the dataset: \n{dataset_sample}")

    # Ask the user to confirm that the data is correct
    data_correct = input("\nDoes the data look correct? (y/n): ")

    # If the data is correct, ask the user to define the columns for setting the training structure
    if data_correct == "y":
        print(f"\nThe available columns are: {dataset_cols}")

        # Ask the user to define the column for the speaker and the column for the quote
        speaker_col = input("Please enter the column for the speaker: ")
        quote_col = input("Please enter the column for the quote: ")

        # show the user the available character speakers
        speakers = available_characters(speaker_col, df)
        print(f"\nThe available characters are: \n{speakers}")

        # Ask the user to define the character to train the model on
        character = input("\nPlease enter the character to train the model on: ")

        # send the character, character column, quote column, and df to the context builder
        context_df = context_builder(character, speaker_col, quote_col, df)

        # ask the user what they would like the model to be called
        model_name = input("\nPlease enter the name you want to use for the model: ")

        # ask the user what Hugging Face repo they would like to use
        repo_name = input(
            "\nPlease enter the name of the Hugging Face repo you would like to use: "
        )

        # send the context_df, model_name, repo_name, and token to the model trainer
        model_tune = fine_tune_model(context_df, model_name, repo_name)

        # show the user the model training results
        print(f"\nHere are the model training results: \n{model_tune}")

    # If the data is not correct, exit the program
    else:
        pass


if __name__ == "__main__":
    main()
