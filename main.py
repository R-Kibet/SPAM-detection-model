import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# READ DATA
data = pd.read_csv(r"/root/Downloads/archive(1)/SPAM text message 20170820 - Data.csv")

print(len(data))
print(data.shape)
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

# EXPLORATORY DATA ANALYSIS
len_ham = len(data['Category'][data.Category == "ham"])
len_spam = len(data['Category'][data.Category == "spam"])

arr = np.array([len_ham, len_spam])
Category = ["ham", "spam"]

print(f"Total number of Ham cases : {len_ham}")
print(f"Total number of spam cases : {len_spam}")

# visualize in a pie chart
plt.pie(arr, labels=Category, explode=[.2, .0], shadow=True)
plt.show()

"""
from the result t wil be an imbalanced data it will learn more ham than spam
"""

# TEXT PROCESSING

"""
since we are dealing with text we need to harmonize them to be of the same  eg  Go and go will be read differently 
"""


def text_process(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'") \
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own") \
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ") \
        .replace("€", " euro ").replace("'ll", " will")
    return x


data["Preprocessed Text"] = data["Message"].apply(lambda x: text_process(x))
print(data.head())

print(data['Message'][0])
print('==================')
print(data['Preprocessed Text'][0])

"""
we can also use stemming ang lemmatization
"""

# FEATURE ENGINEERING
"""
changing the ham and spam to be rep by 0 and 1
"""
data["Category"] = data.Category.map({"ham": 0, "spam": 1})

# DATA DEVELOPMENT AND TRAINING

x_train ,x_test , y_train , y_test = train_test_split( data["Preprocessed Text"], data["Category"],  random_state=1)

print(f'Number of rows in the total set: {format(data.shape[0])}')
print(f'Number of rows in the training set: {format(x_train.shape[0])}')
print(f'Number of rows in the test set: {format(x_test.shape[0])}')


"""
need to convert the txt to numbers so as the model cant read text

countVectorizer , TfidfVectorizer
"""

# instantiate the method
cv = CountVectorizer()

# fit the training data to a matrix
train = cv.fit_transform(x_train)

# Transform testing data an return the matrix : we are not fiiting just transforming
test = cv.transform(x_test)


print(train)


# MODEL DEVELOPMENT
"""
implementing Naive Bayes
"""
NB = MultinomialNB()
NB.fit(train, y_train)

pred = NB.predict(test)

print(pred)

print(f'Accuracy score: ,{ format(accuracy_score(y_test, pred))}')
print(f'Precision score: , {format(precision_score(y_test, pred))}')
print(f'Recall score: , {format(recall_score(y_test, pred))}')
print(f'F1 score: , {format(f1_score(y_test, pred))}')


# TEST THE MODEL
trial = pd.Series("INPUT TEXT HERE")
test = cv.transform(trial)

T =NB.predict(test)
print(T)

