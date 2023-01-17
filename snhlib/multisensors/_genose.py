import json

import pandas as pd


def open_data_genose(item, cols=None):
    if cols is None:
        cols = ["time(s)"] + [f"S{i + 1}" for i in range(10)] + ["Temp", "Humid"]

    if item.find(".csv") != -1:
        data = pd.read_csv(item)
    else:
        data = json.load(open(item, "r"))
        data = data["datasensor"]
        data = pd.DataFrame(data, columns=cols)

    return data
