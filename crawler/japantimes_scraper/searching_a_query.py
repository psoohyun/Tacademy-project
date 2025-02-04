import argparse
import json
import os
import re
from japantimes_scraper import yield_latest_article
from japantimes_scraper import now


def save(json_obj, directory):
    date = json_obj.get("date", "")
    title = re.sub("[^a-zA-Z ]+", "", json_obj.get("title", "")[:50])
    filepath = "{}/{}_{}.json".format(directory, date, title)
    with open(filepath, "w", encoding="utf-8") as fp:
        json.dump(json_obj, fp, indent=2, ensure_ascii=False)


def main():
    today = now()[:10]
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--directory",
        type=str,
        default="C:/Users/13a71/documents/crawling output/japantimes",
        help="Output directory",
    )
    parser.add_argument(
        "--begin_date", type=str, default="2020-01-01", help="begin_date"
    )
    parser.add_argument("--end_date", type=str, default=today, help="end_date")
    parser.add_argument(
        "--section", type=str, default="world", help="newspaper section"
    )
    parser.add_argument(
        "--sleep", type=float, default=1, help="Sleep time for each submission (post)"
    )
    parser.add_argument(
        "--max_page",
        type=int,
        default=100000000,
        help="Number of scrapped articles page",
    )
    parser.add_argument("--verbose", dest="VERBOSE", action="store_true")

    args = parser.parse_args()
    directory = args.directory
    begin_date = args.begin_date
    end_date = args.end_date
    section = args.section
    sleep = args.sleep
    max_num = args.max_page
    VERBOSE = args.VERBOSE

    # check output directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    n_exceptions = 0
    for article in yield_latest_article(section, begin_date, end_date, sleep):
        try:
            save(article, directory)
            print("scraped {}".format(article.get("url"), ""))
        except Exception as e:
            n_exceptions += 1
            print(e)
            continue
        if n_exceptions > 0:
            print("Exist %d article exceptions" % n_exceptions)


if __name__ == "__main__":
    main()
