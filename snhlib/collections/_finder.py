import os
from fnmatch import fnmatch


def get_files(root: str, pattern=["*.csv", "*.json"]) -> list:
    """get_files
    get list of spesific files in spesific path

    Parameters
    ----------
    root : str
        root path directory
    pattern : list, optional
        spesific file type, by default ["*.csv", "*.json"]

    Returns
    -------
    list
        list of spesific files
    """
    list_files = []
    for path, _, files in os.walk(root):
        for name in files:
            if any([fnmatch(name, p) for p in pattern]):
                r = os.path.join(path, name)
                list_files.append(r)
    return list_files
