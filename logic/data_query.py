#!/usr/bin/env python

"""This module will be used to set the parameters for the data used to fine-tune the model."""

import pandas as pd
from datasets import load_dataset


def download_data(data_pathway):
    """This function will take the data_pathway and return a dataframe with the data."""
    dataset = load_dataset(data_pathway)
    df = pd.DataFrame(dataset["train"])
    return df


def dataset_sampler(dataframe):
    """This function will take the dataframe and return a sample and columns."""
    return dataframe.sample(10), dataframe.columns
