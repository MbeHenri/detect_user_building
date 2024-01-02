from pandas import DataFrame

def save_acces_points_2d(access_points, path_csv):
    pacs = DataFrame(access_points).transpose()
    pacs = pacs.reset_index()
    pacs.columns = ["BSSID", "X", "Y"]
    pacs.to_csv(path_csv, sep=";")
    
    return pacs