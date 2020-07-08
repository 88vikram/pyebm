def load_dataset(name=None):
    
    if name==None:
        name = 'EBMData'
    import pandas as pd
    import requests
    import io
    
    if name == 'EBMData':
        urlname = 'https://raw.githubusercontent.com/88vikram/pyebm/master/resources/Data_7.csv'
    elif name == 'CoinitDEBMData':
        urlname = 'https://raw.githubusercontent.com/88vikram/pyebm/master/resources/FixedEffectsData_7.csv'
    s=requests.get(urlname).content
    df=pd.read_csv(io.StringIO(s.decode('utf-8')))
    
    return df