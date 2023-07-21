import argparse
import pandas as pd
import numpy as np
import string as st
import nltk
import re
from nltk import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Query data from linkedin job posts")
    parser.add_argument("--in", dest="input", type=str, required=True, help="Path to input hdf5 data")
    parser.add_argument("--out", dest="output", required=True, type=str, help="Path of output file")

    print("Parsing arguments")
    args = parser.parse_args()
    return args.input, args.output


def list_to_numeric(x):
    if (len(x) == 1):
        return pd.to_numeric(x[0], errors='coerce')
    if (len(x) == 2):
        x = pd.to_numeric(x, errors='coerce')
        # big difference in numbers:
        x.sort()
        if (x[0] < x[1]/1000):
            x[0] = x[0]*1000
        if (x[0] == 0):
            x = x[1]*1000
        x = np.mean(x)
        return x
    else:
        return np.nan


def extract_salary(df):
    s = df['description'].str.extractall(r'([\$][ 0123456789,BMbmilkK+-]*)')
    s = s.reset_index(names=['id', 'match'])
    s = s.rename(columns={0: 'salary_raw'})
    s['salary_str'] = s['salary_raw'].map(lambda x: x.lower().lstrip('$ ').rstrip(' /-.+'))
    s['salary_str'] = s['salary_str'].map(lambda x: x.replace(
        ',', '').replace('.00', '').replace('-', ' ').replace('k', '000'))
    s['salary_str'] = s['salary_str'].map(lambda x: x.split())
    s['salary_numeric'] = s['salary_str'].apply(list_to_numeric)
    return s


def clean_salary(df):
    df.loc[df['salary_numeric'] < 15, 'salary_numeric'] = np.nan
    df.loc[df['salary_numeric'] <= 100, 'salary_numeric'] = df['salary_numeric']*52*40
    df.loc[(df['salary_numeric'] > 100) & (df['salary_numeric'] < 30_000), 'salary_numeric'] = np.nan
    df.loc[df['salary_numeric'] > 1_000_000, 'salary_numeric'] = np.nan
    df = df[~df['salary_numeric'].isna()]
    df = df.groupby(['id']).agg({'salary_numeric': 'mean'}).reset_index()
    return df


def generate_salary(df):
    s = extract_salary(df)
    s = clean_salary(s)
    df = s.join(df, on='id', how='outer').set_index('id').sort_index()
    return df


def remove_punct(text):
    return ("".join([ch for ch in text if ch not in st.punctuation]))


def tokenize(text):
    text = re.split('\s+', text)
    return [x.lower() for x in text]


def remove_small_words(text):
    return [x for x in text if len(x) > 3]


def remove_stopwords(text):
    return [word for word in text if word not in nltk.corpus.stopwords.words('english')]


def stemming(text):
    ps = PorterStemmer()
    return [ps.stem(word) for word in text]


def lemmatize(text):
    word_net = WordNetLemmatizer()
    return [word_net.lemmatize(word) for word in text]


def return_sentences(tokens):
    return " ".join([word for word in tokens])


def clean_text(df, column, lemma, debug=False):
    df['removed_punc'] = df[column].apply(lambda x: remove_punct(x))
    df['tokens'] = df['removed_punc'].apply(lambda msg: tokenize(msg))
    df['larger_tokens'] = df['tokens'].apply(lambda x: remove_small_words(x))
    df['clean_tokens'] = df['larger_tokens'].apply(lambda x: remove_stopwords(x))
    if (lemma):
        df['words'] = df['clean_tokens'].apply(lambda x: lemmatize(x))
    else:
        df['words'] = df['clean_tokens'].apply(lambda wrd: stemming(wrd))
    df['clean_text'] = df['words'].apply(lambda x: return_sentences(x))

    if (debug):
        return df
    df = df.drop(columns=['removed_punc', 'tokens', 'larger_tokens', 'clean_tokens', 'words', column])
    df = df.rename(columns={'clean_text': column})
    return df


def preprocess_text(df):
    df['description'] = df['description'].replace('([\$][ 0123456789,BMbmilkK+-]*)', '', regex=True)
    df['title'] = df['title'].replace('\d+', '', regex=True)
    df = clean_text(df, 'description', lemma=True)
    df = clean_text(df, 'title', lemma=True)
    df = clean_text(df, 'location', lemma=True)
    return df


def clean_dataset(df):
    df = df.drop(columns=['link', 'date', 'company'])
    df = df.rename(columns={'salary_numeric': 'target'})
    df = df[['title', 'location', 'experience', 'description', 'target']]
    df = df[~df['target'].isna()]
    return df


if __name__ == '__main__':
    input, output = parse_arguments()
    read_successful = True

    try:
        df = pd.read_hdf(input)
    except:
        print(f'[ERROR] Cannot open {input}')
        read_successful = False

    if (read_successful):
        df = df.reset_index(drop=True)

        print('Preprocessing file...')
        df = generate_salary(df)
        df = preprocess_text(df)
        df = clean_dataset(df)

        try:
            df.to_csv(output)
            print(f'Successfully saved dataset to {output}')
        except:
            print(f'[ERROR] Cannot save file to {output}')
