import pandas as pd

from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import EventMetrics
from linkedin_jobs_scraper.query import Query, QueryOptions, QueryFilters
from linkedin_jobs_scraper.filters import RelevanceFilters, TypeFilters, ExperienceLevelFilters

EXP_LVL_STR = ['ENTRY_LEVEL', 'ASSOCIATE', 'MID_SENIOR', 'DIRECTOR', 'EXECUTIVE']
EXP_LVL_CLASS = [ExperienceLevelFilters.ENTRY_LEVEL, ExperienceLevelFilters.ASSOCIATE,
                 ExperienceLevelFilters.MID_SENIOR, ExperienceLevelFilters.DIRECTOR, ExperienceLevelFilters.EXECUTIVE]
EXP_LVL_INDEX = 0


def get_scrapper(slow):
    return LinkedinScraper(
        chrome_executable_path='../../chromedriver',  # Custom Chrome executable path (e.g. /foo/bar/bin/chromedriver)
        chrome_options=None,  # Custom Chrome options here
        headless=True,  # Overrides headless mode only if chrome_options is None
        # How many threads will be spawned to run queries concurrently (one Chrome driver for each thread)
        max_workers=1,
        slow_mo=slow,  # Slow down the scraper to avoid 'Too many requests 429' errors (in seconds)
        page_load_timeout=40  # Page load timeout (in seconds)
    )


# Fired once for each page (25 jobs)
def on_metrics(metrics: EventMetrics):
    print('[ON_METRICS]', str(metrics))


def on_error(error):
    print('[ON_ERROR]', error)


def on_end():
    print('[ON_END]')


def run_query(scraper, keyword, location, experience_level, n_queries):
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


def run_queries(scraper, keywords, locations, n_queries, debug):
    for keyword in keywords:
        search_keyword = keyword
        for location in locations:
            for experience_level in range(2, len(EXP_LVL_STR)+2):
                EXP_LVL_INDEX = experience_level-2
                if (debug):
                    print(keyword, location, EXP_LVL_CLASS[EXP_LVL_INDEX], n_queries)
                else:
                    run_query(scraper, keyword, location, EXP_LVL_CLASS[EXP_LVL_INDEX], n_queries)
