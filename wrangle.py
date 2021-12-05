import pandas as pd

def clean_ratings():
    """ 
        Ingests ramen-ratings.csv, 
        Renames a United States value to USA,
        Drops low-count ramen styles Box, Can, and Bar (8 rows),
        Drops countries with less than 5 cumulative observations (29 rows),
        Drops Unrated, nulls and duplicates (16 rows),
        Replaces Stars column with five_stars column,
        Drops 'Review #' and 'Top Ten' columns,
        Renames columns for easier exploration, and
        Returns cleaned dataframe.
    """
    # ingest data
    df = pd.read_csv('ramen-ratings.csv')
    # merge United States and USA
    index_loc = df.loc[df.Country == 'United States'].index.item() # get index
    df.loc[index_loc, 'Country'] = 'USA' # rename United States to USA

    # drop low-count values in package column
    mask = (df.Style == 'Box') | (df.Style == 'Can') | (df.Style == 'Bar')
    df = df[~mask]
    # drop countries with less than 5 cumulative rows in df
    df = drop_low_count_countries(df)
    # drop unrated ramen
    df = df.drop(df.loc[df.Stars == 'Unrated'].index)
    # drop two nulls in Style
    df = df.drop(df[df.Style.isna()].index)

    # change ratings column to bool column showing if rating is five or not
    df = ratings_to_bool(df)
    # drop 'Review #' and 'Top Ten' columns
    df = df.drop(columns=['Review #','Top Ten'])
    # fix remaining column names
    df = df.rename(columns={'Brand':'brand', 
                            'Variety':'product', 
                            'Style':'package', 
                            'Country':'country'})

    # drop 11 duplicate rows
    df = df.drop_duplicates()

    return df

def drop_low_count_countries(df):
    """ Drop a few low-count values in Country from dataframe, return df """
    # get country names for countries with less than 5 cumulative rows in df
    low_count_countries = df.Country.value_counts()[df.Country.value_counts() < 5].index.tolist()
    # init empty index list to extend with indices of low-count countries
    low_count_indices = []
    # iterate through each country having a low value count
    for cntry in low_count_countries:
        # add each index of matching country to list
        low_count_indices.extend(df[df.Country == cntry].index.tolist()) 
    # drop the rows identified as having country with low-count cumulative
    df = df.drop(low_count_indices)

    return df

def ratings_to_bool(df):
    """ Convert ratings column of df to new column specifying whether rating is 5 or not """
    # cast rating column as float
    df['Stars'] = df.Stars.astype('float')
    # new column for 5-star ramen
    df['five_stars'] = df['Stars'] == 5
    # drop rating column
    df = df.drop(columns='Stars')

    return df

