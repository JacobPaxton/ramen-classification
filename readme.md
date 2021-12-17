# Ramen Ratings Classification

![This is an image](https://i.imgur.com/KrCR0jx.jpg)

# Repository Overview
This repository contains my analysis for the Ramen Ratings dataset. The goal is to predict whether a ramen product is rated as five stars or not and identify potential factors that increase or decrease a ramen product's rating using statistical analysis. The final version of this analysis is in **final_notebook.ipynb.**

This repository was my first side project alongside my time in the Codeup data science program. I revisited it a few months later because I wanted to apply the new skills and knowledge I gained since initially working on it to deliver a better product. The previous work remains in the form of old_analysis.ipynb and old_readme.md.

# Link to Data
https://www.kaggle.com/residentmario/ramen-ratings

# Findings
For whether a ramen product will be rated five stars...
1. Ramen brand has little influence
2. Ramen packaging has little influence
3. Ramen country of origin has an observable impact
    * Malaysia, Singapore, and Taiwan have a high ratio of five-star ratings
    * China, Thailand, and USA have a low ratio of five-star ratings
4. Ramen noodle type has little influence
5. Ramen flavor has an observable impact
    * Curry and Sesame ramen products have a high ratio of five-star ratings
    * Chicken and Beef ramen products have a low ratio of five-star ratings
6. Whether the ramen is spicy or not has an observable impact
    * 13% increase in five-star ratings for spicy products
7. Whether the ramen is fried or not has little influence

# Work Summary
## The cool new stuff I accomplished for this project
- **Heavy keyword engineering**
    * Domain research to understand ramen products based on keyword
    * Translation to group all keywords into consistent categories
    * Categorization on common factors based on domain research and translation
- **Multi-layered statistical testing to eliminate features**
    * Chi-Square tests to eliminate initial features that are not related to target
    * One-hot encoding of remaining features' categories
    * Chi-Square tests to eliminate one-hot-encoded categories that are not related to target
- **Clustering country and keyword features into low-, medium-, and high-rate five-star ratings groups**
    * Checked proportions of five-star rating counts against not-five-star rating counts for True in encoded feature
    * Checked proportions of five-star rating counts against not-five-star rating counts for False in encoded feature
    * Compared five-star proportions to check increase/decrease in proportion from False to True
    * Bracketed increasing, middle, and decreasing proportions from False to True
    
## Other stuff that I've done before
- Wrangle
    * Categorize and encode target into five_stars column (classes: is five-stars, isn't five stars)
    * Fix some values, drop some nulls, outliers, and duplicate rows, get rid of unnecessary columns
    * Create univariate visualizations
- Explore
    * Run Chi-Square testing to determine if feature is related to target
    * Feature engineering (overall)
    * Create bivariate visualizations
    * Choose features for model
- Model
    * Choose optimization priorities for the model (F1 Score)
    * Resample the target to address class imbalance
    * Create baseline model and multiple algorithmic models with varying hyperparameter combinations
    * Evaluate models on Validate (first out-of-sample split)
    * Choose best three models in terms of our optimization priority
    * Use GridSearchCV to choose best hyperparameters for top models
    * Calculate ROC AUC of baseline and models
    * Choose best model across all metrics
    * Evaluate baseline and best model on Test split

# Data Dictionary
| Feature             | Datatype   | Description                                |
|:--------------------|:-----------|:-------------------------------------------|
| Review #            | int64      | Unique ID                                  |
| Brand               | object     | Ramen brand                                |
| Variety             | object     | Product name or general description        |
| Style               | object     | Packaging that the ramen product comes in  |         
| Country             | object     | Country that the ramen product comes from  |
| Stars               | object     | Rating from 0 to 5                         | 
| Top Ten             | object     | Year #Rank                                 |       

# Recreate My Work
To re-create this work, follow these steps:
1. Read this README.md
2. Download these files from the repository - ramen-ratings.csv, wrangle.py, explore.py, model.py, and final_notebook.ipynb files into your working directory.
3. To use the SMOTE+Tomek to eliminate class imbalances for train split, you will need to install the tool kit using pip or conda.
   * pip install -U imbalanced-learn
   * conda install -c conda-forge imbalanced-learn
4. Utilize a Jupyter Notebook python3 installation to run final_notebook.ipynb