import pandas as pd

def clean_ratings():
    """ 
        Ingests ramen-ratings.csv, 
        Drops Unrated, nulls and duplicates, 
        Creates new column five_stars, 
        Drops columns Review #, Stars, and Top Ten,
        Renames columns for easier exploration, and
        Returns cleaned dataframe.
    """
    df = pd.read_csv('ramen-ratings.csv')
    # drop unrated ramen
    df = df.drop(df.loc[df.Stars == 'Unrated'].index)
    # cast rating column as float
    df['Stars'] = df.Stars.astype('float')
    # new column for 5-star ramen
    df['five_stars'] = df['Stars'] == 5
    # drop Review #, Stars, and Top Ten columns
    df = df.drop(columns=['Review #','Stars','Top Ten'])
    # drop two nulls in Style
    df = df.drop(df[df.Style.isna()].index)
    # drop 13 duplicate rows
    df = df.drop_duplicates()
    # fix remaining column names
    df = df.rename(columns={'Brand':'brand', 
                            'Variety':'product', 
                            'Style':'package', 
                            'Country':'country'})


    return df