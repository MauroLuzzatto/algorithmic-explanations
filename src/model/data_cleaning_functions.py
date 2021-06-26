from datetime import datetime

import numpy as np


def get_rank_bins(rank):

    bins = [
        (1, 50),
        (51, 100),
        (101, 200),
        (201, 300),
        (301, 400),
        (401, 600),
        (601, 800),
        (801, 3000),
    ]
    # check if the value is a nan
    if rank != rank:
        return "unknown"
    elif "-" in rank:
        value = np.mean([int(value) for value in rank.split("-")])
    else:
        value = int(rank)

    for bin_tuple in bins:
        if bin_tuple[0] <= value <= bin_tuple[1]:
            if bin_tuple[1] == 3000:
                return f"{bin_tuple[0]} and above"
            return f"{bin_tuple[0]}-{bin_tuple[1]}"


def make_date(string):
    return datetime.strptime(string, "%m.%d.%Y")


def get_age(date):
    diff = datetime.today() - date
    return int(diff.days / 365.0)


def replace_line_break(string):
    return string.replace("\r\n", " | ") if isinstance(string, str) else ""
