#!/usr/bin/env python

import pandas as pd


def available_characters(char_col, quotes_df):
    """Returns a list of the available characters in the dataset from the specified column.
    Args:
        char_col (str): The name of the column containing the character names.
    Returns:
        list: Value counts of the available characters knee-capped at 40 lines.
    """

    # get the unique characters
    characters = quotes_df[char_col].value_counts()

    # return the list of characters for those with more than 40 quotes
    return characters[characters > 40]


def context_builder(character, character_col, line_col, quotes_df):
    """This function will take the character, line column, and dataframe and return a list of the character's lines with context.
    Args:
        character (str): The name of the character to train the model on.
        character_col (str): The name of the column containing the character names.
        line_col (str): The name of the column containing the character lines.
        quotes_df (pandas.DataFrame): The dataframe containing the quotes.
    Returns:
        quotes_context_df (pandas.DataFrame): The dataframe containing the quotes with seven preceeding context quotes."""

    # make an empty dataframe to hold the quotes and context
    quotes_context_df = pd.DataFrame(
        columns=[
            "character",
            "quote",
            "context/0",
            "context/1",
            "context/2",
            "context/3",
            "context/4",
            "context/5",
        ]
    )

    # iterate through the quotes dataframe and add the quotes and context to the quotes_context_df starting with row 7
    for i in range(7, len(quotes_df)):
        # if the character in the row matches the character we're looking for
        if quotes_df[character_col][i] == character:
            # add the character, quote, and context to the quotes_context_df
            quotes_context_df = quotes_context_df.append(
                {
                    "character": quotes_df[character_col][i],
                    "quote": quotes_df[line_col][i],
                    "context/0": quotes_df[line_col][i - 1],
                    "context/1": quotes_df[line_col][i - 2],
                    "context/2": quotes_df[line_col][i - 3],
                    "context/3": quotes_df[line_col][i - 4],
                    "context/4": quotes_df[line_col][i - 5],
                    "context/5": quotes_df[line_col][i - 6],
                },
                ignore_index=True,
            )

    # return the quotes_context_df
    return quotes_context_df
