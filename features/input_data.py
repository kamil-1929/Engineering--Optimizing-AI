import pandas as pd
from config.Config import *


def get_input_data() -> pd.DataFrame:
    df1 = pd.read_csv("data//AppGallery.csv", skipinitialspace=True)
    df1.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)
    df2 = pd.read_csv("data//Purchasing.csv", skipinitialspace=True)
    df2.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)
    df = pd.concat([df1, df2], ignore_index=True)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    # df["y"] = df[Config.CLASS_COL]
    # df = df.loc[(df["y2"] != '') & (~df["y2"].isna()),] & df.loc[(df["y3"] != '') & (~df["y3"].isna()),] & df.loc[(df["y4"] != '') & (~df["y4"].isna()),]
    return df
