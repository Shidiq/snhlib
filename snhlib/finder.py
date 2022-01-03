from fnmatch import fnmatch
import os
import pandas as pd
import json


class FinderScan:
    @staticmethod
    def scanroot(root, pattern="*.csv", verbose=0):
        res = []
        for path, subdirs, files in os.walk(root):
            for name in files:
                if fnmatch(name, pattern):
                    r = os.path.join(path, name)
                    res.append(r)
                    if verbose == 1:
                        print(r)
        return res, (path, subdirs, files)

    @staticmethod
    def openDataGeNose(item, cols=None):
        if cols is None:
            cols = ['time(s)'] + \
                [f'S{i + 1}' for i in range(10)] + ['Temp', 'Humid']

        if item.find('.csv') != -1:
            data = pd.read_csv(item)
        else:
            data = json.load(open(item, 'r'))
            data = data['datasensor']
            data = pd.DataFrame(data, columns=cols)

        return data
