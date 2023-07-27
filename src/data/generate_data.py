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


search_keyword = ''
EXP_LVL_STR = ['ENTRY_LEVEL', 'ASSOCIATE', 'MID_SENIOR', 'DIRECTOR', 'EXECUTIVE']
EXP_LVL_CLASS = [ExperienceLevelFilters.ENTRY_LEVEL, ExperienceLevelFilters.ASSOCIATE,
                 ExperienceLevelFilters.MID_SENIOR, ExperienceLevelFilters.DIRECTOR, ExperienceLevelFilters.EXECUTIVE]
EXP_LVL_INDEX = 0


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Query data from linkedin job posts")
    parser.add_argument("--queries", dest="queries", type=str, nargs='+', required=True,
                        help="Keywords to query, example: data, data science, data engineer")
    parser.add_argument("--out", dest="out", required=True,
                        metavar="out", type=str, help="Path of output file, example: ../../data/test.pkl")
    parser.add_argument('--n_queries', dest='n_queries', type=int, help='Number of queries', default=10)
    parser.add_argument('--slow', dest='slow', type=int, help='Number of seconds to wait between queries', default=5)
    parser.add_argument('--locations', dest='locations', type=str, nargs='+', default=["Chicago"],
                        help="Locations where we want to query job posts")
    parser.add_argument('--debug', dest='debug', default=False, action='store_true',
                        help="If debug == True, only print the queries")

    print("Parsing arguments")
    args = parser.parse_args()

    print(args.queries)
    queries = args.queries[0].split(',')
    queries = [q.lstrip() for q in queries]
    return queries, args.locations, args.n_queries, args.out, args.slow, args.debug


def on_data(data: EventData):
    row = [[search_keyword, data.title, data.company, data.link, data.place, data.description,
           data.date, EXP_LVL_STR[EXP_LVL_INDEX]]]
    print(f'Saving job {data.title} job posts to file')
    df = pd.DataFrame(row, columns=['search_keyword', 'title', 'company', 'link', 'location',
                      'description', 'date', 'experience'])
    df.to_hdf(out, key='jobs', append=True, mode='a', format='table', min_itemsize=20000, index=False)


if __name__ == '__main__':

    global out
    keywords, locations, n_queries, out, slow, debug = parse_arguments()

    scraper = get_scrapper(slow)

    # Add event listeners
    scraper.on(Events.DATA, on_data)
    scraper.on(Events.ERROR, on_error)
    scraper.on(Events.END, on_end)

    run_queries(scraper, keywords, locations, n_queries, debug)
