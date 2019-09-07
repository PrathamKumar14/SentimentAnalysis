# Importing the Libraries

import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 


# Loading Datasets

train  = pd.read_csv('train_E6oV3lV.csv')
test   = pd.read_csv('test_tweets_anuFYb8.csv')

# train.head()

# Removing Twitter Handles (@user)
# Combine train and test dataset so that removing @users and many other unwanted words in single step

CombineTrainTest = train.append(test, ignore_index=True)

# Function to remove unwanted words from tweets

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt    

# Creating new column clean_tweet which contain tweets without unwated words
    
CombineTrainTest['Clean_Tweet'] = np.vectorize(remove_pattern)(CombineTrainTest['tweet'], "@[\w]*")

# Remove special characters, numbers, punctuations

CombineTrainTest['Clean_Tweet'] = CombineTrainTest['Clean_Tweet'].str.replace("[^a-zA-Z#]", " ")
 
# Removing short words like hmm, ohh etc

CombineTrainTest['Clean_Tweet'] = CombineTrainTest['Clean_Tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    
CombineTrainTest.head()

# Tokenization

tokenized_tweet = CombineTrainTest['Clean_Tweet'].apply(lambda x: x.split())
tokenized_tweet.head()

# Stemming

from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
tokenized_tweet.head()

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

CombineTrainTest['CLean_Tweet'] = tokenized_tweet

# Understanding the common words used in the tweets: WordCloud

all_words = ' '.join([text for text in CombineTrainTest['Clean_Tweet']])

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Words in non racist/sexist tweets

normal_words =' '.join([text for text in CombineTrainTest['Clean_Tweet'][CombineTrainTest['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Racist/Sexist Tweets

negative_words = ' '.join([text for text in CombineTrainTest['Clean_Tweet'][CombineTrainTest['label'] == 1]])
wordcloud = WordCloud(width=800, height=500,
random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# function to collect hashtags

def hashtag_extract(x):
    hashtags = []
    
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags

# extracting hashtags from non racist/sexist tweets

HT_regular = hashtag_extract(CombineTrainTest['Clean_Tweet'][CombineTrainTest['label'] == 0])

# extracting hashtags from racist/sexist tweets

HT_negative = hashtag_extract(CombineTrainTest['CLean_Tweet'][CombineTrainTest['label'] == 1])

# unnesting list

HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])

#Hashtag plot for non racist/sexist 

a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


#Hashtag plot for racist/sexist 

b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

# Bag-of-word feature

from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(CombineTrainTest['Clean_Tweet'])

# TF-IDF feature

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(CombineTrainTest['Clean_Tweet'])

# Building model using Bag-of-Words features

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model

prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int) # calculating f1 score

# prediction on test set
test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('sub_lreg_bow.csv', index=False) # writing data to a CSV 

# Building model using TF-IDF features

train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

lreg.fit(xtrain_tfidf, ytrain)

prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int)
