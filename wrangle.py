import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------- Main Functions ---------------- #

def prep_explore():
    """ Cleans and splits data for exploration, returns splits """
    df = clean_ratings()
    train, validate, test = split_data(df)
    
    return train, validate, test

def prep_model():
    """ Cleans, engineers, and splits data for modeling, returns splits """
    # clean data
    df = clean_ratings()
    # create country low-, medium- and high-proportion 5-star-review brackets
    df = create_country_features(df)
    # create flavor low-, medium- and high-proportion 5-star-review brackets
    df = create_flavor_features(df)
    # create spicy status feature
    df = create_spicy_feature(df)

    # choose columns
    df = df[['five_stars','is_spicy', 'not_spicy',
             'many_5stars_country','moderate_5stars_country',
             'few_5stars_country','unknown_5stars_country',
             'many_5stars_flavor','moderate_5stars_flavor',
             'few_5stars_flavor','unknown_5stars_flavor']]

    # split data
    train, validate, test = split_data(df)
    # isolate target
    X_train, y_train = train.drop(columns='five_stars'), train.five_stars
    X_validate, y_validate = validate.drop(columns='five_stars'), validate.five_stars
    X_test, y_test = test.drop(columns='five_stars'), test.five_stars

    return X_train, y_train, X_validate, y_validate, X_test, y_test

def split_data(df):
    """ Splits data into Train (60%), Validate (20%), and Test (20%) splits,
        returns splits"""
    train_validate, test = train_test_split(df, test_size=.2, random_state=777)
    train, validate = train_test_split(train_validate, test_size=.25, random_state=777)
    print("Train size:", train.shape, 
          "Validate size:", validate.shape, 
          "Test size:", test.shape)

    return train, validate, test

# ---------------- Prep Explore ---------------- #

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
                            'Variety':'name', 
                            'Style':'package', 
                            'Country':'country'})

    # drop 11 duplicate rows
    df = df.drop_duplicates()

    return df

def drop_low_count_countries(df):
    """ Drops a few low-count values in Country from dataframe, returns df """
    # get country names for countries with less than 5 cumulative rows in df
    counts = df.Country.value_counts()
    low_count_countries = counts[counts < 5].index.tolist()
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
    """ Converts ratings column of df to new column specifying 
        whether rating is 5 or not, returns df """
    # cast rating column as float
    df['Stars'] = df.Stars.astype('float')
    # new column for 5-star ramen
    df['five_stars'] = df['Stars'] == 5
    # drop rating column
    df = df.drop(columns='Stars')

    return df

# ---------------- Prep Model ---------------- #

def create_country_features(df):
    """ Create same rating-driving country features as created in Explore for model """
    # make list of final countries
    final_countries = ['Japan', 'USA', 'South Korea', 'Taiwan', 'China', 'Thailand', 
                       'Malaysia', 'Hong Kong', 'Indonesia', 'Singapore']
    # make three brackets based on 5-star proportion
    high_percent_5star = ['Malaysia', 'Singapore', 'Taiwan']
    mid_percent_5star = ['Hong Kong', 'Japan', 'South Korea', 'Indonesia']
    low_percent_5star = ['China', 'Thailand', 'USA']

    # high bracket feature
    df['many_5stars_country'] = df.country.str.contains('|'.join(high_percent_5star))
    # medium bracket feature
    df['moderate_5stars_country'] = df.country.str.contains('|'.join(mid_percent_5star))
    # low bracket feature
    df['few_5stars_country'] = df.country.str.contains('|'.join(low_percent_5star))
    # unknown bracket feature
    df['unknown_5stars_country'] = df.country.str.contains('|'.join(final_countries)) == False

    return df

def create_flavor_features(df):
    df = create_flavor_values(df)
    """ Create same rating-driving flavor features as created in Explore for model """
    # make list of final flavors
    final_flavors = ['curry', 'chicken', 'crustacean', 'beef', 'sesame', 'pork']
    # make three brackets based on 5-star proportion
    high_percent_5star = ['curry', 'sesame']
    mid_percent_5star = ['pork', 'crustacean']
    low_percent_5star = ['chicken', 'beef']
    # high bracket feature
    df['many_5stars_flavor'] = df.flavor.str.contains('|'.join(high_percent_5star))
    # medium bracket feature
    df['moderate_5stars_flavor'] = df.flavor.str.contains('|'.join(mid_percent_5star))
    # low bracket feature
    df['few_5stars_flavor'] = df.flavor.str.contains('|'.join(low_percent_5star))
    # unknown bracket feature
    df['unknown_5stars_flavor'] = df.flavor.str.contains('|'.join(final_flavors)) == False

    return df

def create_spicy_feature(df):
    """ Create same rating-driving spicy status feature as created in Explore for model """
    # spicy keywords
    keywords = ['Spicy', 'Spice', 'Shin', 'Jjamppong', 'Jjambbong', 'Jjampong', 'Champong', 'Buldalk', 'Buldak', 
                'Sutah', 'Budae', 'RMy', 'Habanero', 'Jinjja', 'Jin', 'Yeul', 'Mala', 'Teumsae', 'Bibim', 
                'Picante', 'Bulnak', 'Volcano', 'Odongtong', 'Sriracha', 'Arrabiata', 'Tom Yum', 'Tom Yam', 
                'tom Yum', 'Tom Saab', 'Tom Klong', 'Suki', 'Laksa', 'Chah Chiang', 'Namja', 'Befikr', 'Mi Goreng', 
                'Kocek', 'Jalapeno', 'Pad Kee Mao', 'Kokomen', 'Wasabi', 'Kung Pao', 'Kimchi', 'Kimchee', 
                'Sabalmyeon', 'Kim Chee', 'Nam Tok', 'Sogokimyun', 'Gentong', 'Chili', 'Chilli', 'chili', 'Cabe', 
                'Yukgaejang', 'Yakisoba', 'Yaki-Soba', 'Yakiosoba']
    # map value
    new_value = 'True'
    # column name
    feature_name = 'is_spicy'
    # map new_value to 'spicy' feature if spicy keyword is in the name
    df = df.apply(lambda row: spicy_mapper(row, keywords, new_value, feature_name), axis=1)

    # non-spicy keywords
    keywords = ['Miso', 'Requeijao', 'Seolleongtang', 'Sukiyaki', 'Jjajangmyeon', 'Jjajangmen', 'Jiajang', 
                'Jjajang', 'Chacharoni', 'Jjawang', 'Ossyoi', 'Batchoy', 'Bajirak', 'Mushroom', 'Shiitake', 
                'Shitake', 'Tomato', 'Clear']
    # map value
    new_value = 'True'
    # column name
    feature_name = 'not_spicy'
    # map new_value to 'spicy' feature if non-spicy keyword is in the name
    df = df.apply(lambda row: spicy_mapper(row, keywords, new_value, feature_name), axis=1)

    # convert column to bool
    df['is_spicy'] = df['is_spicy'] == 'True'
    df['not_spicy'] = df['not_spicy'] == 'True'

    return df

def create_flavor_values(df):
    """ Turn product name keywords into same categories as created in Explore for model """
    # chicken
    keywords = ['Chicken', 'Chikin', 'Duck', 'Pollo', 'Buldalk', 'Buldak', 'Requeijao', 'Gallina']
    new_value = 'chicken'
    df = df.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # beef
    keywords = ['Beef', 'Gomtang', 'Seolleongtang', 'Sukiyaki', 'Nam Tok', 'Sutah', 'Sogokimyun', 
                'Cuchareable', 'Carne', 'Kebab', 'Gentong', 'Bulalo', 'Yukgaejang']
    new_value = 'beef'
    df = df.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # pork
    keywords = ['Pork', 'Prok', 'Jjajangmyeon', 'Jjajangmen', 'Jiajang', 'Jjajang', 'Chacharoni', 
                'Jjawang', 'Tonkotsu', 'Tomkotsu', 'Bacon', 'Ossyoi', 'Yakibuta', 'Batchoy']
    new_value = 'pork'
    df = df.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # crustacean
    keywords = ['Crab', 'Lobster', 'Shrimp', 'Prawn']
    new_value = 'crustacean'
    df = df.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # curry
    keywords = ['Curry', 'curry', 'Betawi', 'Perisa', 'Kari']
    new_value = 'curry'
    df = df.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # veggie
    keywords = ['Clear', 'Veg', 'Oosterse']
    new_value = 'veggie'
    df = df.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # chili
    keywords = ['Chili', 'Chilli', 'chili', 'Cabe']
    new_value = 'chili'
    df = df.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # mushroom
    keywords = ['Mushroom', 'Shiitake', 'Shitake']
    new_value = 'mushroom'
    df = df.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # sesame
    keywords = ['Sesame', 'Sesami']
    new_value = 'sesame'
    df = df.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # chow_mein
    keywords = ['Chow Mein']
    new_value = 'chow_mein'
    df = df.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # kimchi
    keywords = ['Kimchi', 'Kimchee', 'Sabalmyeon', 'Kim Chee']
    new_value = 'kimchi'
    df = df.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # miso
    keywords = ['Miso']
    new_value = 'miso'
    df = df.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # tomato
    keywords = ['Tomato']
    new_value = 'tomato'
    df = df.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # mollusk
    keywords = ['Bajirak', 'Clam', 'Abalone', 'Scallop', 'Vongole']
    new_value = 'mollusk'
    df = df.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # lime
    keywords = ['Lime', 'Jeruk Nipis', 'Kalamansi']
    new_value = 'lime'
    df = df.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)

    return df

def flavor_mapper(row, keywords, new_value):
    """ Map new_value to 'flavor' column based on if a word from keywords exists in product name """
    for word in keywords:
        if word in row['name']:
            row['flavor'] = new_value

    return row

def spicy_mapper(row, keywords, new_value, feature_name):
    """ Map new_value to 'spicy' column based on if a word from keywords exists in product name """
    for word in keywords:
        if word in row['name']:
            row[feature_name] = new_value

    return row