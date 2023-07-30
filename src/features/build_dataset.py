'''
Script to prepare dataset for training

Usage: python build_dataset.py --in ../../data/linkedin_jobs.h5 --out ../../data/preprocessed.zip

'''

import sys
import argparse
import pandas as pd

sys.path.append('../')  # nopep8
from utils.preprocess_utils import generate_salary, preprocess_text, clean_dataset  # nopep8


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Query data from linkedin job posts")
    parser.add_argument("--in", dest="input", type=str, required=True, help="Path to input hdf5 data")
    parser.add_argument("--out", dest="output", required=True, type=str, help="Path of output file")

    print("Parsing arguments")
    args = parser.parse_args()
    return args.input, args.output


if __name__ == '__main__':
    input, output = parse_arguments()
    read_successful = True

    try:
        df = pd.read_hdf(input)
    except Exception as e:
        print(f'[ERROR] Cannot open {input}')
        print(e)
        read_successful = False

    if (read_successful):
        df = df.reset_index(drop=True)

        print('Preprocessing file')
        print('Generating salary...')
        df = generate_salary(df)
        print('Preprocessing text...')
        df = preprocess_text(df)
        print('Cleaning dataset...')
        df = clean_dataset(df)

        try:
            if (output[-3:] == "zip"):
                df.to_csv(output, compression={'method': 'zip', 'archive_name': output.replace('zip', 'csv')})
            elif (output[-3:] == "csv"):
                df.to_csv(output)
            print(f'Successfully saved dataset to {output}')
        except:
            print(f'[ERROR] Cannot save file to {output}')
