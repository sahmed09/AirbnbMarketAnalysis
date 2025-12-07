import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# nltk.download()
mydata = pd.read_csv('Dataset/AB_NYC_2019.csv')
print(mydata.head())

# Dataset Cleaning and Preprocessing
print(mydata.dtypes)

mydata["last_review"] = pd.to_datetime(mydata["last_review"])
print(mydata.dtypes)

print(mydata.isnull().sum())

# mydata['reviews_per_month'].fillna(mydata['reviews_per_month'].mean(), inplace=True)
mydata.fillna({'reviews_per_month': mydata['reviews_per_month'].mean()}, inplace=True)
mydata.drop(columns=['host_name', 'last_review'], axis=1, inplace=True)
print(mydata.isnull().sum())

# Exploratory Data Analysis
# Correlation
numeric_df = mydata.select_dtypes(include=['number'])
corr_1 = numeric_df.corr()
fig, ax = plt.subplots(figsize=(8, 8))
dropSelf = np.zeros_like(corr_1)
dropSelf[np.triu_indices_from(dropSelf)] = True
sns.heatmap(corr_1, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)
plt.savefig("correlation_heatmap.png", dpi=1200, bbox_inches='tight')
plt.show()

"Correlation heatmap shows a high correlation between number of reviews and reviews per month."

plt.figure(figsize=(12, 8))

# Custom color palette (choose your own!)
custom_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

sns.boxplot(
    data=mydata,
    x="neighbourhood_group",
    y="availability_365",
    palette=custom_colors,
    hue="neighbourhood_group"
)

plt.title("Distribution of Availability by Neighborhood Group", fontsize=16)
plt.xlabel("Neighborhood Group", fontsize=14)
plt.ylabel("Availability (365 days)", fontsize=14)
plt.xticks(rotation=30, fontsize=12)

plt.tight_layout()
plt.savefig("availability_boxplot_custom_colors.png", dpi=1200, bbox_inches='tight')
plt.show()

"""
Staten Island Has the Highest Availability Overall. Most listings are available around 200–250 days per year.
Queens and Bronx Show Moderate to High Availability. Bronx also tends toward higher availability, with many listings 
available over 150 days
Manhattan and Brooklyn Have the Lowest Availability. Many listings are available less than 50 days, suggesting high 
occupancy.
"""

plt.figure(figsize=(10, 10))
sns.barplot(data=mydata, x='neighbourhood_group', y='price', hue="neighbourhood_group")
plt.title("Distribution of Price by Neighborhood Group")
plt.xlabel("Neighborhood Group", fontsize=14)
plt.ylabel("Price", fontsize=14)
plt.savefig("Distribution of Price by Neighborhood Group.png", dpi=1200, bbox_inches='tight')
plt.show()

"""Bar plot plotted between neighbourhood group and price shows that Manhattan has the most expensive prices."""

# Price Distribution
df = mydata.copy()
df = df[df['price'].notna()]
df = df[df['price'] > 0]

# Remove extreme outliers (top 1% of price)
price_upper = df['price'].quantile(0.99)
df = df[df['price'] <= price_upper]

plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("Distribution of Price.png", dpi=1200, bbox_inches='tight')
plt.show()


def categorise(hotel_price):
    if hotel_price <= 75:
        return 'Low'
    elif 75 < hotel_price <= 500:
        return 'Medium'
    else:
        return 'High'


mydata['price'].apply(categorise).value_counts().plot(kind='bar')
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.savefig("Distribution of Price2.png", dpi=1200, bbox_inches='tight')
plt.show()

"""Most of the rooms fall in the Medium (75-500 $) category followed by Low and High range rooms. Very few people 
prefer to live in high priced rooms."""

# There are 16 names fields as NaN.Let's replace them with empty string.
mydata['name'].fillna('', inplace=True)
print(mydata['name'].isnull().sum())


# Remove Punctuations
def remove_punctuation_digits_specialchar(line):
    return re.sub('[^A-Za-z]+', ' ', line).lower()


mydata['clean_name'] = mydata['name'].apply(remove_punctuation_digits_specialchar)
# Let's compare raw and cleaned texts.
print(mydata[['name', 'clean_name']].head())


# Remove Stopwords
def tokenize_no_stopwords(line):
    tokens = nltk.tokenize.word_tokenize(line)
    tokens_no_stop = [w for w in tokens if w not in stopwords.words('english')]
    return " ".join(tokens_no_stop)


mydata['final_name'] = mydata['clean_name'].apply(tokenize_no_stopwords)
print(mydata[['clean_name', 'final_name']].head())

# Now let's define a price above 300 as expensive and below 300 as cheap.
# mydata.drop(columns=['target', 'clean_name'], axis=1, inplace=True)
# print(mydata.head())

# Now for using other models, let us first convert the categorical features into numeric by using encoding
le = LabelEncoder()  # Fit label encoder
le.fit(mydata['neighbourhood_group'])
mydata['neighbourhood_group'] = le.transform(mydata['neighbourhood_group'])  # Transform labels to normalized encoding.

le = LabelEncoder()
le.fit(mydata['neighbourhood'])
mydata['neighbourhood'] = le.transform(mydata['neighbourhood'])

le = LabelEncoder()
le.fit(mydata['room_type'])
mydata['room_type'] = le.transform(mydata['room_type'])
print(mydata.head())

# Linear Regression Model
# Prices are not normally distributed as well as there is alot of noise.Hence instead of considering y,
# we consider log(y)
lm = LinearRegression()
mydata = mydata[mydata.price > 0]
mydata = mydata[mydata.availability_365 > 0]

X = mydata[
    ['neighbourhood_group', 'neighbourhood', 'room_type', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
     'calculated_host_listings_count', 'availability_365']]
# Prices are not normally distributed as well as there is alot of noise. Logarithmic conversion of data with huge
# variance can be normalised by logarithmic algorithm.
y = np.log10(mydata['price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lm.fit(X_train, y_train)
y_predicts = lm.predict(X_test)

print("Linear Regression")
print("""
        Mean Squared Error: {}
        R2 Score: {}
        Mean Absolute Error: {}
     """.format(
    np.sqrt(metrics.mean_squared_error(y_test, y_predicts)),
    r2_score(y_test, y_predicts) * 100,
    mean_absolute_error(y_test, y_predicts)
))

# Ridge Model
ridge_model = linear_model.Ridge(alpha=0.01)
ridge_model.fit(X_train, y_train)
y_predicts = ridge_model.predict(X_test)
print("Ridge Regression")
print("""
        Mean Squared Error: {}
        R2 Score: {}
        Mean Absolute Error: {}
     """.format(
    np.sqrt(metrics.mean_squared_error(y_test, y_predicts)),
    r2_score(y_test, y_predicts) * 100,
    mean_absolute_error(y_test, y_predicts)
))

# Lasso Model
Lasso_model = linear_model.Lasso(alpha=0.001)
Lasso_model.fit(X_train, y_train)
y_predicts = Lasso_model.predict(X_test)
print("Lasso Regression")
print("""
        Mean Squared Error: {}
        R2 Score: {}
        Mean Absolute Error: {}
     """.format(
    np.sqrt(metrics.mean_squared_error(y_test, y_predicts)),
    r2_score(y_test, y_predicts) * 100,
    mean_absolute_error(y_test, y_predicts)
))

elastic_model = ElasticNet(alpha=0.001, l1_ratio=0.5)  # l1_ratio controls L1 vs L2 balance
elastic_model.fit(X_train, y_train)
y_predicts = elastic_model.predict(X_test)

print("ElasticNet Regression")
print("""
        Mean Squared Error: {}
        R2 Score: {}
        Mean Absolute Error: {}
     """.format(
    np.sqrt(metrics.mean_squared_error(y_test, y_predicts)),
    r2_score(y_test, y_predicts) * 100,
    mean_absolute_error(y_test, y_predicts)
))

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
)

rf_model.fit(X_train, y_train)
y_predicts = rf_model.predict(X_test)

print("Random Forest Regression")
print("""
        Mean Squared Error: {}
        R2 Score: {}
        Mean Absolute Error: {}
     """.format(
    np.sqrt(metrics.mean_squared_error(y_test, y_predicts)),
    r2_score(y_test, y_predicts) * 100,
    mean_absolute_error(y_test, y_predicts)
))

xgb_model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)

xgb_model.fit(X_train, y_train)
y_predicts = xgb_model.predict(X_test)

print("XGBoost Regression")
print("""
        Mean Squared Error: {}
        R2 Score: {}
        Mean Absolute Error: {}
     """.format(
    np.sqrt(metrics.mean_squared_error(y_test, y_predicts)),
    r2_score(y_test, y_predicts) * 100,
    mean_absolute_error(y_test, y_predicts)
))
