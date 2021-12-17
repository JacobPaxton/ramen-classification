import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------ Chi-Square Functions ------------------------ #

def chi2_ramen_brand(train):
    """ Perform Chi-Square test on Ramen brands, print results """
    # set confidence interval
    alpha = .05
    # create crosstab for chi-square statistical test
    brand_5star_crosstab = pd.crosstab(train.brand, train.five_stars)
    # limit only to brands with sufficient value counts in crosstab 
    # (an assumption of chi-square)
    enough_values_mask = ((brand_5star_crosstab[False] > 5) & 
                          (brand_5star_crosstab[True] > 5))
    # run chi-square test
    _, p, _, _ = stats.chi2_contingency(brand_5star_crosstab[enough_values_mask])
    # check if p is significant
    if p < alpha:
        print("Ramen brand and five-star ratings have a dependent relationship with 95% confidence.")
        print("p-value:", round(p, 3))
    else:
        print("Ramen brand and five-star ratings are independent, did not pass 95% confidence interval.")
        print("p-value:", round(p, 3))

def chi2_ramen_packaging(train):
    """ Perform Chi-Square test on Ramen packagings, print results """
    # set confidence interval
    alpha = .05
    # check dependence of packaging and target
    package_5star_crosstab = pd.crosstab(train.package, train.five_stars)
    # limit only to packaging with sufficient value counts in crosstab 
    # (an assumption of chi-square)
    enough_values_mask = ((package_5star_crosstab[False] > 5) & 
                          (package_5star_crosstab[True] > 5))
    _, p, _, _ = stats.chi2_contingency(package_5star_crosstab)
    # check if p is significant
    if p < alpha:
        print("Packaging and five-star ratings have a dependent relationship with 95% confidence.")
        print("p-value:", round(p, 3))
    else:
        print("Packaging and five-star ratings are independent, did not pass 95% confidence interval.")
        print("p-value:", round(p, 3))

def chi2_ramen_origin_country(train):
    """ Perform Chi-Square test on Ramen country of origin, print results """
    # set confidence interval
    alpha = .05
    # create crosstab for chi-square statistical test
    country_5star_crosstab = pd.crosstab(train.country, train.five_stars)
    # limit only to countries with sufficient value counts in crosstab 
    # (an assumption of chi-square)
    enough_values_mask = ((country_5star_crosstab[False] > 5) & 
                          (country_5star_crosstab[True] > 5))
    # run chi-square test
    _, p, _, _ = stats.chi2_contingency(country_5star_crosstab[enough_values_mask])
    # check if p is significant
    if p < alpha:
        print("Country of origin and five-star ratings have a dependent relationship with 95% confidence.")
        print("p-value:", round(p, 3))
    else:
        print("Country of origin and five-star ratings are independent, did not pass 95% confidence interval.")
        print("p-value:", round(p, 3))

def chi2_ramen_noodle_type(train):
    """ Perform Chi-Square test on engineered Ramen noodle_type column, print results """
    # drop nulls from noodle_type column
    noodle_type_col = train.noodle_type.dropna()
    # set confidence interval
    alpha = .05
    # create crosstab for chi-square statistical test
    noodle_type_5star_crosstab = pd.crosstab(train.noodle_type, train.five_stars)
    # limit only to countries with sufficient value counts in crosstab 
    # (an assumption of chi-square)
    enough_values_mask = ((noodle_type_5star_crosstab[False] > 5) & 
                          (noodle_type_5star_crosstab[True] > 5))
    # run chi-square test
    _, p, _, _ = stats.chi2_contingency(noodle_type_5star_crosstab[enough_values_mask])
    # check if p is significant
    if p < alpha:
        print("Noodle type and five-star ratings have a dependent relationship with 95% confidence.")
        print("p-value:", round(p, 3))
    else:
        print("Noodle type and five-star ratings are independent, did not pass 95% confidence interval.")
        print("p-value:", round(p, 3))

def chi2_ramen_flavor(train):
    """ Perform Chi-Square test on engineered Ramen flavor column, print results """
    # drop nulls from flavor column
    flavor_col = train.flavor.dropna()
    # set confidence interval
    alpha = .05
    # create crosstab for chi-square statistical test
    flavor_5star_crosstab = pd.crosstab(train.flavor, train.five_stars)
    # limit only to countries with sufficient value counts in crosstab 
    # (an assumption of chi-square)
    enough_values_mask = ((flavor_5star_crosstab[False] > 5) & 
                          (flavor_5star_crosstab[True] > 5))
    # run chi-square test
    _, p, _, _ = stats.chi2_contingency(flavor_5star_crosstab[enough_values_mask])
    # check if p is significant
    if p < alpha:
        print("Flavor and five-star ratings have a dependent relationship with 95% confidence.")
        print("p-value:", round(p, 3))
    else:
        print("Flavor and five-star ratings are independent, did not pass 95% confidence interval.")
        print("p-value:", round(p, 3))

def chi2_ramen_spicy_status(train):
    """ Perform Chi-Square test on engineered Ramen spicy column, print results """
    # drop nulls from spicy column
    spicy_col = train.spicy.dropna()
    # set confidence interval
    alpha = .05
    # create crosstab for chi-square statistical test
    spicy_5star_crosstab = pd.crosstab(train.spicy, train.five_stars)
    # limit only to countries with sufficient value counts in crosstab 
    # (an assumption of chi-square)
    enough_values_mask = ((spicy_5star_crosstab[False] > 5) & 
                          (spicy_5star_crosstab[True] > 5))
    # run chi-square test
    _, p, _, _ = stats.chi2_contingency(spicy_5star_crosstab[enough_values_mask])
    # check if p is significant
    if p < alpha:
        print("Spicy status and five-star ratings have a dependent relationship with 95% confidence.")
        print("p-value:", round(p, 3))
    else:
        print("Spicy status and five-star ratings are independent, did not pass 95% confidence interval.")
        print("p-value:", round(p, 3))

def chi2_ramen_fried_status(train):
    """ Perform Chi-Square test on engineered Ramen fried column, print results """
    # drop nulls from fried column
    fried_col = train.fried.dropna()
    # set confidence interval
    alpha = .05
    # create crosstab for chi-square statistical test
    fried_5star_crosstab = pd.crosstab(train.fried, train.five_stars)
    # limit only to countries with sufficient value counts in crosstab 
    # (an assumption of chi-square)
    enough_values_mask = ((fried_5star_crosstab[False] > 5) & 
                          (fried_5star_crosstab[True] > 5))
    # run chi-square test
    _, p, _, _ = stats.chi2_contingency(fried_5star_crosstab[enough_values_mask])
    # check if p is significant
    if p < alpha:
        print("Fried status and five-star ratings have a dependent relationship with 95% confidence.")
        print("p-value:", round(p, 3))
    else:
        print("Fried status and five-star ratings are independent, did not pass 95% confidence interval.")
        print("p-value:", round(p, 3))

# ------------------------ Mapping Functions ------------------------ #

def noodle_type_mapper(row, keywords, new_value):
    """ Map keywords in specified list to specified category in 'noodle_type' """
    for word in keywords:
        if word in row['name']:
            row['noodle_type'] = new_value

    return row

def flavor_mapper(row, keywords, new_value):
    """ Map keywords in specified list to specified category in 'flavor' """
    for word in keywords:
        if word in row['name']:
            row['flavor'] = new_value

    return row

def spicy_mapper(row, keywords, new_value):
    """ Map keywords in specified list to specified category in 'spicy' """
    for word in keywords:
        if word in row['name']:
            row['spicy'] = new_value

    return row

def fried_mapper(row, keywords, new_value):
    """ Map keywords in specified list to specified category in 'fried' """
    for word in keywords:
        if word in row['name']:
            row['fried'] = new_value

    return row

# ------------------------ Feature Creators ------------------------ #

def create_has_keyword(train):
    """ Create column indicating if row's product name contains a keyword or not """
    # identify all keywords, prepare list for df.col.str.contains()
    keyword_mask = '|'.join(['Vermicelli', 'Vernicalli', 'Bihun', 'Sano', 'Chicken', 
    'Chikin', 'Duck', 'Vegetable', 'Veggie', 'Vegetarian','Beef', 'Gomtang', 
    'Seolleongtang', 'Sukiyaki', 'Nam Tok', 'Pork', 'Jjajangmen', 'Jiajang', 
    'Tonkotsu', 'Tomkotsu', 'Bacon', 'Budae', 'Seafood', 'Crab', 'Anchovy', 'Bajirak', 
    'Clam', 'Abalone', 'Scallop', 'Vongole', 'Salmon', 'Lobster', 'Shrimp', 'Prawn', 
    'Tuna', 'Tteok', 'Rabokki', 'Raobokki', 'Spicy', 'Spice', 'Shin', 'Jjamppong', 
    'Jjambbong', 'Buldalk', 'Sutah', 'Budae', 'Habanero', 'Jinjja', 'Jin', 'Yeul', 
    'Mala', 'Teumsae', 'Bibim', 'Picante', 'Bulnak', 'Volcano', 'Odongtong', 
    'Sriracha', 'Arrabiata', 'Tom Yum', 'Tom Yam', 'Tom Saab', 'Tom Klong', 'Suki', 
    'Stir Fry', 'Bokkeum', 'Tteokbokki', 'Topokki', 'Yukgaejang', 'Rabokki', 
    'Yakisoba', 'Yaki-Soba', 'Yakiosoba', 'Fried', 'Non-Fried', 'Goreng', 
    'Ramyonsari', 'Keopnurungji', 'Sabalmyeon', 'Miso', 'Teriyaki', 'Mushroom', 'Udon', 
    'Udoin', 'Tomato', 'Chili', 'Chilli', 'chili', 'Wonton', 'Wantan', 'Pickled', 
    'Sesame', 'Superior', 'Carbonara', 'Chow Mein', 'Sweet', 'Pad Thai', 'Sour', 
    'sour', 'Curry', 'Soy', 'Shoyu', 'Shiitake', 'Shitake', 'Tofu', 'Pho', 'Clear', 
    'Egg', 'Tempura', 'Laksa', 'Buckwheat', 'Soba', 'Salt', 'Shio', 'Sio', 'Tomato', 
    'Neapolitan', 'Napolitan', 'Spaghetti', 'Mayo', 'Barbecue', 'BBQ', 'Masala', 
    'Kimchi', 'Veg','Tteobokki', 'Rice', 'Onion', 'Pollo', 'Cheese', 'Betawi', 
    'Chah Chiang','Namja', 'Perisa', 'Kari', 'Jjawang', 'Jjajangmyeon', 'Sogokimyun', 
    'Jjajang', 'Ossyoi', 'Befikr', 'curry', 'Sotanghon', 'U-Dong', 'U-dong', 
    'Mi Goreng', 'Kocok', 'Chacharoni', 'Yakibuta', 'Cuchareable', 'RMy', 'Jalapeno', 
    'Biryani', 'Carne', 'Kimchee', 'Pad Kee Mao', 'Kalguksoo', 'Prok', 'Nipis', 
    'Jjampong', 'Buldak', 'tom Yum', 'Sesami', 'Kim Chee', 'Kebab', 'Hyoubanya', 
    'Batchoy', 'Gentong', 'Kokomen', 'Requeijao', 'Champong', 'Gallina', 'Bulalo', 
    'Wasabi', 'Kalamansi', 'Cabe', 'Oosterse', 'Kung Pao'])
    # create True/False for whether the row contains a keyword in the product name
    train['has_keyword'] = train.name.str.contains(keyword_mask)

    return train

def check_non_keywords(train):
    """ Check value counts of words I did not designate as keywords in product names """
    # check all rows without keywords for each unique word's value counts in entire list
    print(
        pd.Series( # make a Series of each instance of each word
            ' '.join(
                    train[~train.has_keyword]    # look at rows we haven't caught with a keyword yet
                    .name.tolist()        # put all 'name' cells in a list
                    ).split()        # join all lists into one string, then split the string into a list of each word
        ).value_counts()        # calculate the value counts of each word in the series
        .head(10)         # display the top 10 (changed from 30 to 10 after the words I wanted were captured)
    )

def create_noodle_type(train):
    """ 
        Group Ramen noodle-related keywords in Ramen product names into categories, 
        Assign categories to new column called 'noodle_type',
        Return dataframe with new column. 
    """
    # rice
    keywords = ['Rice', 'Vermicelli', 'Vernicalli', 'Bihun', 'Biryani', 'Tteokbokki', 
                'Tteobokki', 'Topokki', 'Rabokki']
    new_value = 'rice'
    train = train.apply(lambda row: noodle_type_mapper(row, keywords, new_value), axis=1)
    # wheat
    keywords = ['Udon', 'Udoin', 'U-Dong', 'U-dong', 'Sano', 'Spaghetti', 'Carbonara', 
                'Neapolitan', 'Napolitan', 'Kalguksoo']
    new_value = 'wheat'
    train = train.apply(lambda row: noodle_type_mapper(row, keywords, new_value), axis=1)
    # buckwheat
    keywords = ['Buckwheat', 'Soba']
    new_value = 'buckwheat'
    train = train.apply(lambda row: noodle_type_mapper(row, keywords, new_value), axis=1)

    return train

def create_flavor(train):
    """ 
        Group Ramen flavor-related keywords in Ramen product names into categories, 
        Assign categories to new column called 'flavor',
        Return dataframe with new column.
    """
    # chicken
    keywords = ['Chicken', 'Chikin', 'Duck', 'Pollo', 'Buldalk', 'Buldak', 'Requeijao', 
                'Gallina']
    new_value = 'chicken'
    train = train.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # beef
    keywords = ['Beef', 'Gomtang', 'Seolleongtang', 'Sukiyaki', 'Nam Tok', 'Sutah', 
                'Sogokimyun', 'Cuchareable', 'Carne', 'Kebab', 'Gentong', 'Bulalo', 
                'Yukgaejang']
    new_value = 'beef'
    train = train.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # pork
    keywords = ['Pork', 'Prok', 'Jjajangmyeon', 'Jjajangmen', 'Jiajang', 'Jjajang', 
                'Chacharoni', 'Jjawang', 'Tonkotsu', 'Tomkotsu', 'Bacon', 'Ossyoi', 
                'Yakibuta', 'Batchoy']
    new_value = 'pork'
    train = train.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # crustacean
    keywords = ['Crab', 'Lobster', 'Shrimp', 'Prawn']
    new_value = 'crustacean'
    train = train.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # curry
    keywords = ['Curry', 'curry', 'Betawi', 'Perisa', 'Kari']
    new_value = 'curry'
    train = train.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # veggie
    keywords = ['Clear', 'Veg', 'Oosterse']
    new_value = 'veggie'
    train = train.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # chili
    keywords = ['Chili', 'Chilli', 'chili', 'Cabe']
    new_value = 'chili'
    train = train.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # mushroom
    keywords = ['Mushroom', 'Shiitake', 'Shitake']
    new_value = 'mushroom'
    train = train.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # sesame
    keywords = ['Sesame', 'Sesami']
    new_value = 'sesame'
    train = train.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # chow_mein
    keywords = ['Chow Mein']
    new_value = 'chow_mein'
    train = train.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # kimchi
    keywords = ['Kimchi', 'Kimchee', 'Sabalmyeon', 'Kim Chee']
    new_value = 'kimchi'
    train = train.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # miso
    keywords = ['Miso']
    new_value = 'miso'
    train = train.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # tomato
    keywords = ['Tomato']
    new_value = 'tomato'
    train = train.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # mollusk
    keywords = ['Bajirak', 'Clam', 'Abalone', 'Scallop', 'Vongole']
    new_value = 'mollusk'
    train = train.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)
    # lime
    keywords = ['Lime', 'Jeruk Nipis', 'Kalamansi']
    new_value = 'lime'
    train = train.apply(lambda row: flavor_mapper(row, keywords, new_value), axis=1)

    return train

def create_spicy(train):
    """ 
        Group Ramen spicy-related keywords in Ramen product names into categories, 
        Assign categories to new column called 'spicy',
        Return dataframe with new column.
    """
    # is spicy
    keywords = ['Spicy', 'spicy', 'Spice', 'Shin', 'Jjamppong', 'Jjambbong', 'Jjampong', 
                'Champong', 'Buldalk', 'Buldak', 'Sutah', 'Budae', 'RMy', 'Habanero', 
                'Jinjja', 'Jin', 'Yeul', 'Mala', 'Teumsae', 'Bibim', 'Picante', 'Bulnak', 
                'Volcano', 'Odongtong', 'Sriracha', 'Arrabiata', 'Tom Yum', 'Tom Yam', 
                'tom Yum', 'Tom Saab', 'Tom Klong', 'Suki', 'Laksa', 'Chah Chiang', 
                'Namja', 'Befikr', 'Mi Goreng', 'Kocek', 'Jalapeno', 'Pad Kee Mao', 
                'Kokomen', 'Wasabi', 'Kung Pao', 'Kimchi', 'Kimchee', 'Sabalmyeon', 
                'Kim Chee', 'Nam Tok', 'Sogokimyun', 'Gentong', 'Chili', 'Chilli', 
                'chili', 'Cabe', 'Yukgaejang', 'Yakisoba', 'Yaki-Soba', 'Yakiosoba']
    new_value = 'True'
    train = train.apply(lambda row: spicy_mapper(row, keywords, new_value), axis=1)
    # isn't spicy
    keywords = ['Miso', 'Requeijao', 'Seolleongtang', 'Sukiyaki', 'Jjajangmyeon', 
                'Jjajangmen', 'Jiajang', 'Jjajang', 'Chacharoni', 'Jjawang', 'Ossyoi', 
                'Batchoy', 'Bajirak', 'Mushroom', 'Shiitake', 'Shitake', 'Tomato', 
                'Clear']
    new_value = 'False'
    train = train.apply(lambda row: spicy_mapper(row, keywords, new_value), axis=1)

    return train

def create_fried(train):
    """ 
        Group Ramen fried-related keywords in Ramen product names into categories, 
        Assign categories to new column called 'fried',
        Return dataframe with new column.
    """
    # fried
    keywords = ['Stir Fry', 'Stir-Fried', ' Fried', 'Bokkeum', 'Tteokbokki', 
                'Tteobokki', 'Topokki', 'Yukgaejang', 'Rabokki', 'Yakisoba', 
                'Yaki-Soba', 'Yakiosoba', 'Goreng', 'Tempura', 'Kung Pao', 
                'Sukiyaki', 'Kebab', 'Gentong', 'Bulalo', 'Jjajangmyeon', 'Jjajangmen', 
                'Jiajang', 'Jjajang', 'Chacharoni', 'Jjawang', 'Tonkotsu', 'Tomkotsu', 
                'Bacon', 'Yakibuta', 'Batchoy', 'Chow Mein']
    new_value = 'True'
    train = train.apply(lambda row: fried_mapper(row, keywords, new_value), axis=1)
    # not fried
    keywords = ['Non-Fried', 'Requeijao', 'Yakisoba', 'Yaki-Soba', 'Yakiosoba', 
                'Gomtang', 'Seolleongtang', 'Nam Tok', 'Sutah', 'Sogokimyun', 
                'Cuchareable', 'Gomtang', 'Yukgaejang', 'Ossyoi', 'Clear']
    new_value = 'False'
    train = train.apply(lambda row: fried_mapper(row, keywords, new_value), axis=1)

    return train

# ------------------------ Country vs Five-Star Reviews ------------------------ #

def country_five_star_counts(train):
    """ Check total review counts and five-star review counts for each country """
    # init empty list for results
    results_list = []
    # iterate each country and check results
    for cntry in train.country.unique().tolist():
        # get total count of country's reviews
        total_count = (train.country == cntry).sum()
        # get count of 5-star reviews for country
        count_5stars = (train[train.five_stars == 'True'].country == cntry).sum()
        # check if at least 5 5-star reviews
        enough = count_5stars >= 5
        # get proportion between 5-star reviews and total count
        proportion_5stars = count_5stars / total_count
        # add results to list
        results_list.append(f'{cntry} ({enough}): {total_count} reviews, {int(proportion_5stars * 100)}% 5-star')
        
    return results_list

def country_five_star_features(train):
    """ 
        Create high-, medium-, and low-proportion features for 
        country against five-star ratings.
    """
    # make list of final countries
    final_countries = ['Japan', 'USA', 'South Korea', 'Taiwan', 'China', 'Thailand', 
                       'Malaysia', 'Hong Kong', 'Indonesia', 'Singapore']
    # bracket countries into high-, medium-, and low-proportion five star review brackets
    high_percent_5star = ['Malaysia', 'Singapore', 'Taiwan']
    mid_percent_5star = ['Hong Kong', 'Japan', 'South Korea', 'Indonesia']
    low_percent_5star = ['China', 'Thailand', 'USA']
    # high bracket
    train['many_5stars_country'] = train.country.str.contains('|'.join(high_percent_5star))
    # medium bracket
    train['moderate_5stars_country'] = train.country.str.contains('|'.join(mid_percent_5star))
    # low bracket
    train['few_5stars_country'] = train.country.str.contains('|'.join(low_percent_5star))
    # unknown bracket
    train['unknown_5stars_country'] = train.country.str.contains('|'.join(final_countries)) == False
    # drop redundant country column
    train.drop(columns='country', inplace=True)

    return train

# ------------------------ Flavor vs Five-Star Reviews ------------------------ #

def flavor_five_star_counts(train):
    non_null_flavor = train[train.flavor.isna() == False]
    # init empty list for results
    results_list = []
    # iterate each flavor and check results
    for flvr in non_null_flavor.flavor.unique().tolist():
        # get total count of flavor's reviews
        total_count = (non_null_flavor.flavor == flvr).sum()
        # get count of 5-star reviews for flavor
        count_5stars = (non_null_flavor[non_null_flavor.five_stars == 'True'].flavor == flvr).sum()
        # check if at least 5 5-star reviews
        enough = count_5stars >= 5
        # get proportion between 5-star reviews and total count
        proportion_5stars = count_5stars / total_count
        # add results to list
        results_list.append(f'{flvr} ({enough}): {total_count} reviews, {int(proportion_5stars * 100)}% 5-star')
    
    return results_list

def flavor_five_star_features(train):
    """ 
        Create high-, medium-, and low-proportion features for 
        flavor against five-star ratings.
    """
    # make list of final flavors
    final_flavors = ['curry', 'chicken', 'crustacean', 'beef', 'sesame', 'pork']
    # make three brackets based on 5-star proportion
    high_percent_5star = ['curry', 'sesame']
    mid_percent_5star = ['pork', 'crustacean']
    low_percent_5star = ['chicken', 'beef']
    # high bracket, turn nulls to false (captured in 'unknown_5stars_flavor)
    train['many_5stars_flavor'] = train.flavor.str.contains('|'.join(high_percent_5star))
    train['many_5stars_flavor'] = train['many_5stars_flavor'].fillna(False)
    # medium bracket, turn nulls to false (captured in 'unknown_5stars_flavor)
    train['moderate_5stars_flavor'] = train.flavor.str.contains('|'.join(mid_percent_5star))
    train['moderate_5stars_flavor'] = train['moderate_5stars_flavor'].fillna(False)
    # low bracket, turn nulls to false (captured in 'unknown_5stars_flavor)
    train['few_5stars_flavor'] = train.flavor.str.contains('|'.join(low_percent_5star))
    train['few_5stars_flavor'] = train['few_5stars_flavor'].fillna(False)
    # unknown bracket
    train['unknown_5stars_flavor'] = train.flavor.str.contains('|'.join(final_flavors)) == False
    # drop redundant flavor column
    train.drop(columns='flavor', inplace=True)

    return train

# ------------------------ Visualizations ------------------------ #

def univariate_bars(train):
    """ Show histograms for the Ramen features post-keyword categorization """
    # initial univariate histograms
    sns.histplot(train.five_stars) # five_stars
    plt.title('Target: five_stars')
    plt.xlabel('')
    plt.show()

    sns.histplot(train.country) # country
    plt.title('Feature: country')
    plt.xticks(rotation=90)
    plt.xlabel('')
    plt.show()

    sns.histplot(flavor_col) # flavor
    plt.title('Feature: flavor')
    plt.xticks(rotation=60)
    plt.xlabel('')
    plt.show()

    sns.histplot(spicy_col) # spicy
    plt.title('Feature: spicy')
    plt.xlabel('')
    plt.show()

def bivariate_bars(train):
    """ 
        Show separate histograms for is-five-stars and not-five-stars for 
        each model-ready feature.
    """
    # iterate through each non-target feature
    for col in train.columns[1:]:
        # create samples based on 5-star and not 5-star reviews
        col_not_5star = train[train.five_stars == 'False'][col].dropna().astype('str')
        col_is_5star = train[train.five_stars == 'True'][col].dropna().astype('str')
        # viz header
        print('-'*20, col, '-'*20)
        print(train[col].value_counts())
        # visualization
        plt.subplot(121) # Observations without five_stars (left charts)
        plt.title('Not Five Stars')
        sns.histplot(col_not_5star.sort_values(ascending=False))
        plt.subplot(122) # Observations with five_stars (right charts)
        plt.title('Five Stars')
        sns.histplot(col_is_5star.sort_values(ascending=False), color='#ffa500')
        plt.tight_layout()
        plt.show()
        not_5star_ratio = (col_not_5star == "True").sum() / col_not_5star.shape[0]
        is_5star_ratio = (col_is_5star == "True").sum() / col_is_5star.shape[0]
        percent_difference = is_5star_ratio / not_5star_ratio
        print(f'Not 5-Star t/f ratio: {int(not_5star_ratio * 100)}%')
        print(f'Is 5-Star t/f ratio: {int(is_5star_ratio * 100)}%')
        print(f'Percent Difference of Is 5-Star / Not 5-Star: {int(percent_difference * 100)}%')