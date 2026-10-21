#%%
import sys
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

import argparse

parser = argparse.ArgumentParser(description='Add the path to your csv file that you want to check')
parser.add_argument('-p','--path', help='Relative path to csv file', required=True)
# parser.add_argument("-ew", "--exclude_weather", help="Set this to True if you do not want to check for the weather variables")
args = vars(parser.parse_args())
# args = {"path": "./test_corrupt.csv"}
print(args)
print("Loading file...", args["path"])

EXPECTED_COLS = ["forecast_date", "target", "horizon", "q0.025", "q0.25",
                 "q0.5", "q0.75", "q0.975"]
LEN_EXP_COLS = len(EXPECTED_COLS)

# if bool(args["exclude_weather"]) == True:
#     print("Excluding weather variables!")
#     TARGETS = ["DAX", "energy"]
# else:
TARGETS = ["demand", "res_demand", "temperature", "wind"]

TARGET_VALS = dict(demand = [str(i) + " hour" for i in (36,40,44,60,64,68)],
                   res_demand = [str(i) + " hour" for i in (36,40,44,60,64,68)],
                   temperature = [str(i) + " hour" for i in (36,40,44,60,64,68)],
                   wind = [str(i) + " hour" for i in (36,40,44,60,64,68)]
                   )

TARGET_LEN = dict(demand = len(TARGET_VALS['demand']),
                  res_demand = len(TARGET_VALS['res_demand']),
                  temperature = len(TARGET_VALS['temperature']),
                  wind = len(TARGET_VALS['wind'])
                  )

TARGET_PLAUS = dict(demand = [0,200],
                    res_demand = [-200,200],
                    temperature = [-50,50],
                    wind = [0, 70])


COLS_QUANTILES = ["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]


def check_df(path):

    try:
        df = pd.read_csv(path, sep=",")
    except FileNotFoundError:
        print("File not found.")
        print("Stopping early...")
        sys.exit()
    except pd.errors.EmptyDataError:
        print("No data in given file.")
        print("Stopping early...")
        sys.exit()
    except pd.errors.ParserError:
        print("Parse error.")
        print("Stopping early...")
        sys.exit()

    print("File loaded! Start Checking...")
    print("---------------------------")
    col_names = df.columns


    #%%
    print("Checking the Columns...")
    # Check column length
    if len(col_names) != LEN_EXP_COLS:
        print("Dataset contains ",len(col_names), "columns. Required are",LEN_EXP_COLS)
        print("Stopping early...")
        sys.exit()

    if set(col_names) != set(EXPECTED_COLS):
        print("Dataset does not contain the required columns (or more).")
        missing_cols = set(EXPECTED_COLS) - set(col_names)
        print("The missing columns are:", missing_cols)
        print("Stopping early...")
        sys.exit()

    for i,col in enumerate(EXPECTED_COLS):
        if col == col_names[i]:
            continue
        else:
            print("Columns not in correct order. Order should be:", EXPECTED_COLS)
            print("Your order is:", col_names.values)
            print("Stopping early...")
            sys.exit()

    #%%
    # Date Col
    print("Checking type of columns...")
    try:
        df["forecast_date"] = pd.to_datetime(df["forecast_date"], format="%Y-%m-%d",
                                            errors="raise")
    except (pd.errors.ParserError, ValueError):
        print("Could not parse Date in format YYYY-MM-DD")
        print("Stopping early...")
        sys.exit()

    try:
        df["target"] = df["target"].astype("object", errors="raise")
    except ValueError:
        print("Cannot convert target column to String.")
        print("Stopping early...")
        sys.exit()

    try:
        df["horizon"] = df["horizon"].astype("object", errors="raise")
    except ValueError:
        print("Cannot convert horizon column to String.")
        print("Stopping early...")
        sys.exit()

    #for cq in COLS_QUANTILES:
    df[COLS_QUANTILES] = df[COLS_QUANTILES].apply(
        pd.to_numeric, errors="coerce"
    )

    #if df[COLS_QUANTILES].isna().any(axis=None):
    #    print("Some quantile values are missing or not numeric!")
    #    print("Stopping early...")
    #    sys.exit(1)
    nan_mask = df[COLS_QUANTILES].isna()

    if nan_mask.any(axis=None):
        first_idx = nan_mask.any(axis=1).idxmax()
        first_quantile = nan_mask.loc[first_idx].idxmax()

        print(
            f"Some quantile values are missing or not numeric! "
            f"First occurrence: target '{df.loc[first_idx, 'target']}', "
            f"horizon '{df.loc[first_idx, 'horizon']}', "
            f"quantile '{first_quantile}'."
        )
        print("Stopping early...")
        sys.exit(1)

    # Reject infinity
    if np.isinf(df[COLS_QUANTILES]).any(axis=None):
        print("Quantile values cannot be infinite.")
        print("Stopping early...")
        sys.exit(1)

    # %%
    print("Checking if the Dates make sense...")

    if len(pd.unique(df["forecast_date"])) > 1:
        print("forecast_date needs to be the same in all rows.")
        print("Stopping early...")
        sys.exit()

    if df["forecast_date"][0].date() < datetime.today().date():
        print("----WARNING: Forecast date should not be in the past.")
        print("")
        # warnings.warn("Forecast date should not be in the past.")

    if df["forecast_date"][0].weekday() != 2:
        print("----WARNING: Forecast date should be a Wednesday.")
        print("")
        # warnings.warn("Forecast date should be a Wednesday")



    print("Checking targets...")

    if not df["target"].isin(TARGETS).all():
        print(f"Target column can only contain {TARGETS}. Check spelling.")
        print("Stopping early...")
        sys.exit()

    for target in TARGETS:

        if len(df[df["target"] == target]) != TARGET_LEN[target]:
            print(f"Exactly {TARGET_LEN[target]} rows need to have target = {target}")
            #if target == "demand":
            #    print("Exactly 6 rows need to have target = ", target)
            #else:
            #    print("Exactly 5 rows need to have target =", target)
            print("Stopping early...")
            sys.exit()

        if (df[df["target"] == target]["horizon"] != TARGET_VALS[target]).any():
            print("Target", target, "horizons need to be (in this order):", TARGET_VALS[target])
            print("Stopping early...")
            sys.exit()

        if (df[df["target"] == target][COLS_QUANTILES] < TARGET_PLAUS[target][0]).any(axis=None) or \
            (df[df["target"] == target][COLS_QUANTILES] > TARGET_PLAUS[target][1]).any(axis=None):
            print("----WARNING: Implausible values for",target,"detected. You may want to re-check.")
            print("")
            # warnings.warn("Implausible values for "+str(target)+" detected. You may want to re-check them.")

    print("Checking quantiles...")

#    ALL_NAN_IDX = df[df.isna().any(axis=1)].index
#   NAN_TARGET_IDX_LIST = []

#    if len(ALL_NAN_IDX) != 0:
#        NAN_TARGET = df.iloc[ALL_NAN_IDX[0]]["target"]


#        NAN_TARGET_IDX_LIST = df[df["target"] == NAN_TARGET].index

#        print(f"You have NaN-values in target {NAN_TARGET}...")
#        print("Stopping early...")
#        sys.exit()

    for i, row in df.iterrows():
        #if i in NAN_TARGET_IDX_LIST:
        #    continue

        diffs = row[COLS_QUANTILES].diff()
        if diffs[1:].isna().any():
            print("Something is wrong with your quantiles.")
            print("Stopping early...")
            sys.exit()
        diffs.iloc[0] = 0
        if (diffs < 0).any():
            print("Predictive quantiles in row",i,"are not ordered correctly (need to be non-decreasing)")
            print("Stopping early...")
            sys.exit()

    print("---------------------------")
    print("Looks good!")


check_df(args["path"])