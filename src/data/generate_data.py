'''
Script to download recent job offers from linkedin based on keyword given by the user

Usage: 
python generate_data.py --queries "data analyst, data scientist, data engineer" --out ../../data/test.h5 --n_queries 10
'''


import argparse
import pandas as pd
import logging
from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import Events, EventData, EventMetrics
from linkedin_jobs_scraper.query import Query, QueryOptions, QueryFilters
from linkedin_jobs_scraper.filters import RelevanceFilters, TypeFilters, ExperienceLevelFilters


# Change root logger level (default is WARN)
logging.basicConfig(level=logging.INFO)

EXP_LVL_STR = ['ENTRY_LEVEL', 'ASSOCIATE', 'MID_SENIOR', 'DIRECTOR', 'EXECUTIVE']
EXP_LVL_CLASS = [ExperienceLevelFilters.ENTRY_LEVEL, ExperienceLevelFilters.ASSOCIATE,
                 ExperienceLevelFilters.MID_SENIOR, ExperienceLevelFilters.DIRECTOR, ExperienceLevelFilters.EXECUTIVE]
EXP_LVL_INDEX = 0

search_keyword = ''


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


# Fired once for each page (25 jobs)
def on_metrics(metrics: EventMetrics):
    print('[ON_METRICS]', str(metrics))


def on_error(error):
    print('[ON_ERROR]', error)


def on_end():
    print('[ON_END]')


def run_queries(keyword, location, experience_level, n_queries):
    queries = [
        Query(
            query=keyword,
            options=QueryOptions(
                locations=[location],
                apply_link=False,
                skip_promoted_jobs=True,
                page_offset=0,
                limit=n_queries,
                filters=QueryFilters(
                    relevance=RelevanceFilters.RECENT,
                    type=[TypeFilters.FULL_TIME],
                    experience=[experience_level]
                )
            )
        ),
    ]

    scraper.run(queries)


if __name__ == '__main__':

    global out
    keywords, locations, n_queries, out, slow, debug = parse_arguments()

    scraper = LinkedinScraper(
        chrome_executable_path='../../chromedriver',  # Custom Chrome executable path (e.g. /foo/bar/bin/chromedriver)
        chrome_options=None,  # Custom Chrome options here
        headless=True,  # Overrides headless mode only if chrome_options is None
        # How many threads will be spawned to run queries concurrently (one Chrome driver for each thread)
        max_workers=1,
        slow_mo=slow,  # Slow down the scraper to avoid 'Too many requests 429' errors (in seconds)
        page_load_timeout=40  # Page load timeout (in seconds)
    )

    # Add event listeners
    scraper.on(Events.DATA, on_data)
    scraper.on(Events.ERROR, on_error)
    scraper.on(Events.END, on_end)

    for keyword in keywords:
        search_keyword = keyword
        for location in locations:
            for experience_level in range(2, len(EXP_LVL_STR)+2):
                EXP_LVL_INDEX = experience_level-2
                if (debug):
                    print(keyword, location, EXP_LVL_CLASS[EXP_LVL_INDEX], n_queries)
                else:
                    run_queries(keyword, location, EXP_LVL_CLASS[EXP_LVL_INDEX], n_queries)
