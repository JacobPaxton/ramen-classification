import pandas as pd
from scipy import stats

# ------------------------ Chi-Square Functions ------------------------ #

def chi2_ramen_brand(train):
    """ Perform Chi-Square test on Ramen brands, print results """
    # set confidence interval
    alpha = .05
    # create crosstab for chi-square statistical test
    brand_5star_crosstab = pd.crosstab(train.brand, train.five_stars)
    # limit only to brands with sufficient value counts in crosstab 
    # (an assumption of chi-square)
    enough_values_mask = (brand_5star_crosstab[False] > 5) & 
                         (brand_5star_crosstab[True] > 5)
    # run chi-square test
    _, p, _, _ = stats.chi2_contingency(brand_5star_crosstab[enough_values_mask])
    # check if p is significant
    if p < alpha:
        print("Ramen brand and five-star ratings have a dependent relationship with 95% confidence.")
        print("p-value:", p)
    else:
        print("Ramen brand and five-star ratings are independent, did not pass 95% confidence interval.")
        print("p-value:", p)

def chi2_ramen_packaging(train):
    """ Perform Chi-Square test on Ramen packagings, print results """
    # set confidence interval
    alpha = .05
    # check dependence of packaging and target
    package_5star_crosstab = pd.crosstab(train.package, train.five_stars)
    # limit only to packaging with sufficient value counts in crosstab 
    # (an assumption of chi-square)
    enough_values_mask = (package_5star_crosstab[False] > 5) & 
                         (package_5star_crosstab[True] > 5)
    _, p, _, _ = stats.chi2_contingency(package_5star_crosstab)
    # check if p is significant
    if p < alpha:
        print("Packaging and five-star ratings have a dependent relationship with 95% confidence.")
        print("p-value:", p)
    else:
        print("Packaging and five-star ratings are independent, did not pass 95% confidence interval.")
        print("p-value:", p)

def chi2_ramen_origin_country(train):
    """ Perform Chi-Square test on Ramen country of origin, print results """
    # set confidence interval
    alpha = .05
    # create crosstab for chi-square statistical test
    country_5star_crosstab = pd.crosstab(train.country, train.five_stars)
    # limit only to countries with sufficient value counts in crosstab 
    # (an assumption of chi-square)
    enough_values_mask = (country_5star_crosstab[False] > 5) & 
                         (country_5star_crosstab[True] > 5)
    # run chi-square test
    _, p, _, _ = stats.chi2_contingency(country_5star_crosstab[enough_values_mask])
    # check if p is significant
    if p < alpha:
        print("Country of origin and five-star ratings have a dependent relationship with 95% confidence.")
        print("p-value:", p)
    else:
        print("Country of origin and five-star ratings are independent, did not pass 95% confidence interval.")
        print("p-value:", p)

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
    enough_values_mask = (noodle_type_5star_crosstab[False] > 5) & 
                         (noodle_type_5star_crosstab[True] > 5)
    # run chi-square test
    _, p, _, _ = stats.chi2_contingency(noodle_type_5star_crosstab[enough_values_mask])
    # check if p is significant
    if p < alpha:
        print("Noodle type and five-star ratings have a dependent relationship with 95% confidence.")
        print("p-value:", p)
    else:
        print("Noodle type and five-star ratings are independent, did not pass 95% confidence interval.")
        print("p-value:", p)

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
    enough_values_mask = (flavor_5star_crosstab[False] > 5) & 
                         (flavor_5star_crosstab[True] > 5)
    # run chi-square test
    _, p, _, _ = stats.chi2_contingency(flavor_5star_crosstab[enough_values_mask])
    # check if p is significant
    if p < alpha:
        print("Flavor and five-star ratings have a dependent relationship with 95% confidence.")
        print("p-value:", p)
    else:
        print("Flavor and five-star ratings are independent, did not pass 95% confidence interval.")
        print("p-value:", p)

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
    enough_values_mask = (spicy_5star_crosstab[False] > 5) & 
                         (spicy_5star_crosstab[True] > 5)
    # run chi-square test
    _, p, _, _ = stats.chi2_contingency(spicy_5star_crosstab[enough_values_mask])
    # check if p is significant
    if p < alpha:
        print("Spicy status and five-star ratings have a dependent relationship with 95% confidence.")
        print("p-value:", p)
    else:
        print("Spicy status and five-star ratings are independent, did not pass 95% confidence interval.")
        print("p-value:", p)

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
    enough_values_mask = (fried_5star_crosstab[False] > 5) & 
                         (fried_5star_crosstab[True] > 5)
    # run chi-square test
    _, p, _, _ = stats.chi2_contingency(fried_5star_crosstab[enough_values_mask])
    # check if p is significant
    if p < alpha:
        print("Fried status and five-star ratings have a dependent relationship with 95% confidence.")
        print("p-value:", p)
    else:
        print("Fried status and five-star ratings are independent, did not pass 95% confidence interval.")
        print("p-value:", p)

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
    keywords = ['Fried', 'Stir Fry', 'Bokkeum', 'Tteokbokki', 'Tteobokki', 'Topokki', 
                'Yukgaejang', 'Rabokki', 'Yakisoba', 'Yaki-Soba', 'Yakiosoba', 'Goreng', 
                'Tempura', 'Kung Pao', 'Sukiyaki', 'Kebab', 'Gentong', 'Bulalo', 
                'Jjajangmyeon', 'Jjajangmen', 'Jiajang', 'Jjajang', 'Chacharoni', 
                'Jjawang', 'Tonkotsu', 'Tomkotsu', 'Bacon', 'Yakibuta', 'Batchoy', 
                'Chow Mein']
    new_value = 'True'
    train = train.apply(lambda row: fried_mapper(row, keywords, new_value), axis=1)
    # not fried
    keywords = ['Non-Fried', 'Requeijao', 'Yakisoba', 'Yaki-Soba', 'Yakiosoba', 
                'Gomtang', 'Seolleongtang', 'Nam Tok', 'Sutah', 'Sogokimyun', 
                'Cuchareable', 'Gomtang', 'Yukgaejang', 'Ossyoi', 'Clear']
    new_value = 'False'
    train = train.apply(lambda row: fried_mapper(row, keywords, new_value), axis=1)

    return train