"""The main control function for the model trainer."""

from logic.data_query import download_data, dataset_sampler
from logic.data_setup import available_characters, context_builder

from sklearn.model_selection import train_test_split


def data_setup():
    """Function to collect user input at desired intervals and return outputs"""

    # show a welcome message
    print(
        "\nWelcome to this Hugging Face GPT-Medium model trainer. \nIt will give you step-by-step prompts to prepare your data for fine-tuning. \nA couple things to note: spelling counts, and you need to use Hugging Face to accomplish this."
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

        # print a quick sample and confirm it looks right
        print(f"\nHere is a sample of the context: \n{context_df}")
        context_correct = input("\nDoes the context look correct? (y/n): ")

        # if the context is correct, return the context df
        if context_correct == "y":

            trn_df, val_df = train_test_split(context_df, test_size=0.2)

            return trn_df, val_df

        # if the context is not correct, return to the beginning of the function
        else:
            data_setup()

    # If the data is not correct, exit the program
    else:
        data_setup()


if __name__ == "__main__":
    data_setup()
