import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from nltk.stem.porter import *
import numpy as np
from imblearn.over_sampling import RandomOverSampler

# from nltk.classify import maxent
# import datetime

stemmer = PorterStemmer()
f = open('output.txt', 'w')

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

print("Creating Bag-Of-words\n")
#trainFile = pd.read_csv("corpus/input.csv")
trainFile = pd.read_csv("corpus/train_lyrics_1000.csv")

#testFile = pd.read_csv("testFile.csv")
testFile = pd.read_csv("valid_lyrics_200.csv")

f.write("Input file:\n")
f.write(str(testFile))

i = 0
for raw_song in testFile["lyrics"]:
    letters_only = re.sub("[^\D]", " ", raw_song)  # The text to search
    testFile["lyrics"][i] = letters_only
    i += 1

# artist
i = 0
for raw_song_artist in testFile["artist"]:
    letters_only = re.sub("[^\D]", " ", raw_song_artist)  # The text to search
    testFile["artist"][i] = letters_only
    i += 1

# mood
i = 0
for raw_song_mood in testFile["mood"]:
    letters_only = re.sub("[^\D]", " ", raw_song_mood)  # The text to search
    testFile["mood"][i] = letters_only
    i += 1

analyzer = CountVectorizer().build_analyzer()
print("Creating Word vectors\n")
vectorizer = CountVectorizer(analyzer=stemmed_words,
                             tokenizer=None,
                             lowercase=True,
                             preprocessor=None,
                             max_features=5000
                             )

train_data_features = vectorizer.fit_transform([r for r in trainFile["lyrics"]]) # If we work on not changed, we need to put "song"
test_data_features = vectorizer.transform([r for r in testFile["lyrics"]]) # If we work on not changed, we need to put "song"

# For mood
train_data_features_mood = vectorizer.fit_transform([r for r in trainFile["mood"]])
test_data_features_mood = vectorizer.transform([r for r in testFile["mood"]])

# For artist
train_data_features_artist = vectorizer.fit_transform([r for r in trainFile["artist"]])
test_data_features_artist = vectorizer.transform([r for r in testFile["artist"]])

train_data_features = train_data_features.toarray()
test_data_features = test_data_features.toarray()

train_data_features_mood = train_data_features_mood.toarray()
test_data_features_mood = test_data_features_mood.toarray()

train_data_features_artist = train_data_features_artist.toarray()
test_data_features_artist = test_data_features_artist.toarray()

print("Resampling corpus\n")
rs = RandomOverSampler()
X_resampledRe, y_resampledRe = rs.fit_sample(train_data_features,trainFile["genre"])

print("fitting for Naive bayes\n")
clf = MultinomialNB()
clf.fit(train_data_features, trainFile["genre"])
f.write("\nOutput from Naive Bayes Multi Normal:\n")
predicted = clf.predict(test_data_features)

f.write(str(predicted))
f.write("\naccuracy: ")
f.write(str(np.mean(predicted == testFile['genre'])))
f.write("\n")

print("fitting for Naive bayes with resampled corpus\n")
clf = MultinomialNB()
clf.fit(X_resampledRe, y_resampledRe)
f.write("\nOutput from Naive Bayes Multi RE:\n")
predicted = clf.predict(test_data_features)

f.write(str(predicted))
f.write("\naccuracy: ")
f.write(str(np.mean(predicted == testFile['genre'])))
f.write("\n")

print("fitting for SVC\n")
clf = SVC()
clf.fit(train_data_features, trainFile["genre"])
f.write("\nOutput from SVC Normal:\n")
predicted = clf.predict(test_data_features)
f.write(str(predicted))
f.write("\naccuracy: ")
f.write(str(np.mean(predicted == testFile['genre'])))
f.write("\n")

print("fitting for SVC with resampled corpus\n")
clf = SVC()
clf.fit(X_resampledRe, y_resampledRe)
f.write("\nOutput from SVC RE:\n")
predicted = clf.predict(test_data_features)
f.write(str(predicted))
f.write("\naccuracy: ")
f.write(str(np.mean(predicted == testFile['genre'])))
f.write("\n")


# Mood

print("Resampling corpus for mood\n")
rs = RandomOverSampler()
X_resampledRe, y_resampledRe = rs.fit_sample(train_data_features_mood, trainFile["mood"])

print("fitting for Naive bayes for mood\n")
clf = MultinomialNB()
clf.fit(train_data_features_mood, trainFile["mood"])
f.write("\nOutput from Naive Bayes Multi Normal for mood:\n")
predicted = clf.predict(test_data_features_mood)

f.write(str(predicted))
f.write("\naccuracy: ")
f.write(str(np.mean(predicted == testFile['mood'])))
f.write("\n")

print("fitting for Naive bayes with resampled corpus for mood\n")
clf = MultinomialNB()
clf.fit(X_resampledRe, y_resampledRe)
f.write("\nOutput from Naive Bayes Multi RE for mood:\n")
predicted = clf.predict(test_data_features_mood)

f.write(str(predicted))
f.write("\naccuracy: ")
f.write(str(np.mean(predicted == testFile['mood'])))
f.write("\n")

print("fitting for SVC for mood\n")
clf = SVC()
clf.fit(train_data_features_mood, trainFile["mood"])
f.write("\nOutput from SVC Normal for mood:\n")
predicted = clf.predict(test_data_features_mood)
f.write(str(predicted))
f.write("\naccuracy: ")
f.write(str(np.mean(predicted == testFile['mood'])))
f.write("\n")

print("fitting for SVC with resampled corpus for mood\n")
clf = SVC()
clf.fit(X_resampledRe, y_resampledRe)
f.write("\nOutput from SVC RE:\n")
predicted = clf.predict(test_data_features_mood)
f.write(str(predicted))
f.write("\naccuracy: ")
f.write(str(np.mean(predicted == testFile['mood'])))
f.write("\n")

# Artist

print("Resampling corpus for artist\n")
rs = RandomOverSampler()
X_resampledRe, y_resampledRe = rs.fit_sample(train_data_features_artist, trainFile["artist"])

print("fitting for Naive bayes for artist\n")
clf = MultinomialNB()
clf.fit(train_data_features_artist, trainFile["artist"])
f.write("\nOutput from Naive Bayes Multi Normal for artist:\n")
predicted = clf.predict(test_data_features_artist)

f.write(str(predicted))
f.write("\naccuracy: ")
f.write(str(np.mean(predicted == testFile['mood'])))
f.write("\n")

print("fitting for Naive bayes with resampled corpus for artist\n")
clf = MultinomialNB()
clf.fit(X_resampledRe, y_resampledRe)
f.write("\nOutput from Naive Bayes Multi RE for artist:\n")
predicted = clf.predict(test_data_features_artist)

f.write(str(predicted))
f.write("\naccuracy: ")
f.write(str(np.mean(predicted == testFile['artist'])))
f.write("\n")

print("fitting for SVC for artist\n")
clf = SVC()
clf.fit(train_data_features_artist, trainFile["artist"])
f.write("\nOutput from SVC Normal for artist:\n")
predicted = clf.predict(test_data_features_artist)
f.write(str(predicted))
f.write("\naccuracy: ")
f.write(str(np.mean(predicted == testFile['artist'])))
f.write("\n")

print("fitting for SVC with resampled corpus for artist\n")
clf = SVC()
clf.fit(X_resampledRe, y_resampledRe)
f.write("\nOutput from SVC RE:\n")
predicted = clf.predict(test_data_features_artist)
f.write(str(predicted))
f.write("\naccuracy: ")
f.write(str(np.mean(predicted == testFile['artist'])))
f.write("\n")

print("Completed!\nCheck output.txt for results")
