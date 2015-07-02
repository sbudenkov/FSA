#The aim is to use an NLTK naive Bayes classifier to classify
#movie reviews as either positive or negative using the Movie Reviews Dataset

from nltk.corpus import movie_reviews #get movie reviews dataset (1k pos, 1k neg)
import nltk
import random

def main():

    #get each doc and corresponding doc class (pos or neg)
    docs = [(list(movie_reviews.words(fileid)), category) \
    for category in movie_reviews.categories() \
    for fileid in movie_reviews.fileids(category)]

    #check number of docs in dataset (should be 2000)
    print ("Number of docs in dataset is: ")
    print (len(movie_reviews.fileids()))

    #check classes for docs in dataset (should be 'pos' and 'neg')
    print ("Classes available: ")
    print (movie_reviews.categories())

    #mix up the docs
    random.shuffle(docs)

    #list the frequency distribution of words and convert to lowercase
    words_in_dataset = nltk.FreqDist(words.lower() for words in movie_reviews.words())

    #get the words to use as features (top 2000 most frequent words only)
    word_features = list(words_in_dataset)[:2000]

    #for each doc, for each word feature, check if it is included in doc,
    #and return a binary value to represent included or not included in doc
    def get_features(doc):
        doc_words = set(doc)
        features = {}
        for word in word_features:
            features['contains(%s)' % word] = (word in doc_words)
        return features

    #save features for each doc in each class
    featuresets = [(get_features(_doc), _class) for (_doc,_class) in docs]

    #give test set 200 docs (10%) and train set 1800 docs (90%)
    train_set, test_set = featuresets[200:], featuresets[:200]
    print ("Docs in training dataset: ", len(train_set))
    print ("Docs in test dataset: ", len(test_set))

    #give train set to NB classifier
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    #print accuracy
    print("Final accuracy is: ")
    print(nltk.classify.accuracy(classifier, test_set))

    #print(classifier.show_most_informative_features(10))

if __name__ == '__main__':
    main()