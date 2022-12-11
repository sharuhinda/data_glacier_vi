import logging
import os
import pandas as pd
import yaml
import re
import sys

# Setting up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(logging.Formatter(fmt='[%(asctime)s: %(funcName)s: %(levelname)s] %(message)s'))
logger.addHandler(handler)

def get_config(file):
    try:
        with open(file, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as exc:
        logger.error(exc)
        return None


def re_replace(string, pattern, repl):
    return re.sub(pattern, repl, string)


def validate_columns(df, config):
    cols_expected = list(config['cols_expected'])
    cols_expected = [x.lower() for x in cols_expected]
    cols_output = list(config['cols_validated'])
    
    map_col_names = {cols_expected[i]: cols_output[i] for i in range(len(cols_expected))}

    cols_received = df.columns.str.lower()
    cols_received = [x.lower().strip() for x in cols_received] # lower case and strip leading and trailing whitespaces
    cols_received = [re.sub(r'\s+', r'_', x) for x in cols_received] # replacing 1 or more whitespaces in the middle of column name by '_'

    df.columns = cols_received

    cols_valid = [x for x in cols_received if x in cols_expected] # check if cols_valid are in list of expected columns
    cols_to_drop = [x for x in cols_received if x not in cols_expected]
    # check if number of found columns equals number of expected columns (but not sure that all columns are presented)
    if set(cols_valid) != set(cols_expected):
        logger.error('Validation FAILED: Not all expected columns found! Missing columns: ', set(cols_expected).difference(set(cols_valid)))
        return 0

    # replace received column names by valid ones based on map dictionary
    for i, colname in enumerate(cols_valid):
        cols_valid[i] = map_col_names[colname]

    df.drop(columns=cols_to_drop, inplace=True)
    df.columns = cols_valid
    logger.debug('Validation PASSED: all expected columns found')
    return 1
