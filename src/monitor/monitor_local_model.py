'''
Script to download recent job offers from linkedin based on keyword given by the user

Usage: 
python generate_data.py --queries "data analyst, data scientist, data engineer" --out ../../data/test.h5 --n_queries 10
'''

import sys
import argparse
import pandas as pd
from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import Events, EventData
from linkedin_jobs_scraper.filters import ExperienceLevelFilters

sys.path.append('../')  # nopep8
from utils.data_utils import on_error, on_end, run_queries, get_scrapper  # nopep8
from utils.preprocess_utils import generate_salary, preprocess_text, clean_dataset  # nopep8


search_keyword = ''
EXP_LVL_STR = ['ENTRY_LEVEL', 'ASSOCIATE', 'MID_SENIOR', 'DIRECTOR', 'EXECUTIVE']
EXP_LVL_CLASS = [ExperienceLevelFilters.ENTRY_LEVEL, ExperienceLevelFilters.ASSOCIATE,
                 ExperienceLevelFilters.MID_SENIOR, ExperienceLevelFilters.DIRECTOR, ExperienceLevelFilters.EXECUTIVE]
EXP_LVL_INDEX = 0


def parse_arguments():
    parser = argparse.ArgumentParser(description="Query data from linkedin job posts")

    print("Parsing arguments")
    args = parser.parse_args()


def on_data(data: EventData):
    row = [[search_keyword, data.title, data.company, data.link, data.place, data.description,
           data.date, EXP_LVL_STR[EXP_LVL_INDEX]]]
    df = pd.DataFrame(row, columns=['search_keyword', 'title', 'company', 'link', 'location',
                      'description', 'date', 'experience'])
    df = generate_salary(df)
    df = preprocess_text(df)
    df = clean_dataset(df, dropna=False)
    print(df[['title', 'target']])


if __name__ == '__main__':

    scraper = get_scrapper(slow=5)

    # Add event listeners
    scraper.on(Events.DATA, on_data)
    scraper.on(Events.ERROR, on_error)
    scraper.on(Events.END, on_end)

    run_queries(scraper, ['data scientist', 'data engineer', 'data analyst'], ["Chicago"], 10, debug=False)
