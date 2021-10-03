# Repository Overview
My analysis of Kaggle dataset 'Ramen-Ratings'. Produce insights and build a classification model that predicts ratings of ramen based on brand, style, variety, and country of origin.

# Link to Data
https://www.kaggle.com/residentmario/ramen-ratings

# Highlights, Takeaways
- Decided to use a classification approach to predict ramen ratings (star ratings from 0 to 5)
- Conducted a lot of statistical testing against existing and engineered features
- Created multiple engineered features
- Created many models with varying algorithms and hyperparameters
- Modified precision of target for modeling
- Beat baseline prediction accuracy with multiple models and feature combinations

# Specific Work That Was Done
0. Decide how to approach data
    * Rating is dependent variable
    * First try to predict each 0.25-step rating increment between 0 and 5
    * Then try to predict each 1.0-step rating increment between 0 and 5
    * Engineer columns having too many unique values into new columns
1. Ingest data
    * Download data from kaggle.com
    * Read data using pandas.read_csv()
2. Tidy up data for analysis/modeling
    * Fix dtypes
    * Eliminate unrated observations
    * Remove values with counts too low
    * Remove nulls
    * Check distributions
3. Explore data
    * Statistical testing of package styles
        * Bowls rank statistically higher than cups (95% confidence)
    * Statistical testing of country of origin
        * Japan statistically produces better-rated ramen than USA. (95% confidence)
        * Japan statistically produces better-rated ramen than the rest. (95% confidence)
    * Statistical testing of brands
        * Nissin does not statistically produce better-rated ramen than the rest.
    * Analysis of 5-star reviews by country
    * Analysis of 5-star reviews by brand
    * Engineer several features based on Variety
        * Artificial, Instant, Chicken, Beef, Shrimp, Seafood, Chow Mein, Spicy, Veggie
    * Statistical testing on all nine engineered features
        * Statistically higher rating than rest: None
        * Statistically lower rating than rest: Chicken, Beef, Shrimp, Veggie
        * Not statistically higher or lower: Seafood, Chow Mein, Spicy
4. Create machine learning models on 0.25-step rating increment
    * Calculate mean baseline
    * Build minimum viable product (MVP) model using Brand, Style, Country
        * Encode values for modeling
        * Split data for in-sample and out-of-sample testing with a set random_state
        * Isolate target (rating) from data
        * Create and fit models on in-sample data using varying hyperparameters
            * Random Forest, Decision Tree, K-Nearest Neighbors, Logistic Regression
        * Make predictions on in-sample and out-of-sample data
        * Compare accuracy of models against baseline for both data samples
            * Classification Report shows not all values had over-0% prediction accuracy
            * Performed best: DecisionTreeClassifier with max_depth=16, min_samples_leaf=4 (beats baseline)
    * Build models for data where the Brand has more than 10 reviews
        * Limit dataframe on Brand
        * Follow same steps as building MVP
        * Performed best: LogisticRegression with l2 penalty (beats baseline)
    * Build models using engineered Variety features
        * Create new columns for the features (see: Explore data section)
        * Follow same steps as building MVP
        * Performed best: LogisticRegression without l2 penalty (beats baseline)
    *  Use Recursive Feature Elimination to determine best features
        * RFE identifies one-hot-encoded Country column as better than engineered Variety and also one-hot-encoded Style
    * Create models using only one-hot-encoded Country as a feature
        * Performed best: LogisticRegression without l2 penalty (beats baseline)
5. Create machine learning models on 1.0-step rating increment
    * Decrease target precision from 0.25-step to 1.0-step
    * Calculate mean baseline
    * Build models using previous feature combinations (each approach replicated)
        * Only Country best-performed: KNearestClassifier with 60 neighbors
        * Brand >10 best-performed: DecisionTreeClassifier with max_depth=10, min_samples_leaf=1
        * Engineered features *and* Style, Country best-performed: DecisionTreeClassifier with max_depth=2, min_samples_leaf=1
        * Only-Country and Add-Brand approaches had better prediction accuracy than baseline, but All-Possible performed equally with baseline

# Recreate my work
1. Download the data from the link above
2. Initialize a Jupyter Notebook server in the same directory as the data
3. Run all cells in ramen_ratings_analysis.ipynb