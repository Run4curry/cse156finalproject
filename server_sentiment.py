#!/bin/python
import sys
def read_files(tarfname):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
            
            
    class Data: pass
    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    #sentiment.count_vect = CountVectorizer(max_features=14000,ngram_range=(1,3))
    # best
    sentiment.tfidf_vect = TfidfVectorizer(max_features=15000,ngram_range=(1,3),sublinear_tf=True, binary=True)
    # second best
    #sentiment.tfidf_vect = TfidfVectorizer(max_features=13000,ngram_range=(1,3))
    #third best
    #sentiment.tfidf_vect = TfidfVectorizer()


    #sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    #sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    sentiment.trainX = sentiment.tfidf_vect.fit_transform(sentiment.train_data)
    print(sentiment.trainX.shape)
    sentiment.devX = sentiment.tfidf_vect.transform(sentiment.dev_data)
    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment

def read_unlabeled(tarfname, sentiment):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the sentiment.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name
            
    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)
        
            
    #unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    unlabeled.X = sentiment.tfidf_vect.transform(unlabeled.data)

    print(unlabeled.X.shape)
    tar.close()
    return unlabeled

def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels

def write_pred_kaggle_file(unlabeled, cls, outfname, sentiment):
    """Writes the predictions in Kaggle format.

    Given the unlabeled object, classifier, outputfilename, and the sentiment object,
    this function write sthe predictions of the classifier on the unlabeled data and
    writes it to the outputfilename. The sentiment object is required to ensure
    consistent label names.
    """
    yp = cls.predict(unlabeled.X)
    labels = sentiment.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    for i in range(len(unlabeled.data)):
        f.write(str(i+1))
        f.write(",")
        f.write(labels[i])
        f.write("\n")
    f.close()

def first_classification_task(unlabeled, cls, sentiment):
    import numpy as np
    yp = cls.predict(unlabeled.X)
    scores = cls.predict_proba(unlabeled.X)
    labels = sentiment.le.inverse_transform(yp)
    
    coefficients = cls.coef_[0]
    k = 40
    top_k =np.argsort(coefficients)[-k:]
    top_k_words = []

    print('-'*50)
    print('Top k=%d' %k)
    print('-'*50)

    for i in top_k:
        print(sentiment.tfidf_vect.get_feature_names()[i])
        top_k_words.append(sentiment.tfidf_vect.get_feature_names()[i])
    #print(sentiment.count_ve
    print('-'*50)
    print('Bottom k=%d' %k)
    print('-'*50)
    #top_k = np.argpartition(coefficients, -k)[-k:]
    bottom_k =np.argsort(coefficients)[:k]
    bottom_k_words = []
    #print(top_k)
    for i in bottom_k:
        print(sentiment.tfidf_vect.get_feature_names()[i])
        bottom_k_words.append(sentiment.tfidf_vect.get_feature_names()[i])
    
    import nltk
    from nltk.corpus import stopwords
    stopwords = set(stopwords.words('english'))

    top_set = set()
    bottom_set = set()
    # add the positive words to the positive sets
    for i in range(len(top_k_words)):
        top_set.add(top_k_words[i])
    # add the negative words to the negative sets
    for i in range(len(bottom_k_words)):
        bottom_set.add(bottom_k_words[i])
    for i in range(len(labels)):
        # confidence postive prediction
   #     print("---------------------------------------------------------")
    #    print(unlabeled.data[i])
        #print(scores[i])
        #print(labels[i])
        if labels[i] == "POSITIVE" and scores[i][1] >= 0.70:
            result = []
            for word in unlabeled.data[i].split():
                if word in top_set:
                    result.append(word)

    #        print("This sentence is positive because of these words")
            for word in result:
                if word not in stopwords:
                    print(word)
    #        print("The probability of it being positive is", scores[i][1])

        elif labels[i] == "NEGATIVE" and scores[i][0] >= 0.70:
            result = []
            for word in unlabeled.data[i].split():
                if word in bottom_set:
                    result.append(word)

        #    print("This sentence is negative because of these words")
            for word in result:
                if word not in stopwords or word == 'not' or word == 'but':
                    print(word)
         #   print("The probability of it being negative is", scores[i][0])

        else:
            pos_set = []
            neg_set = []
            for word in unlabeled.data[i].split():
                if word in top_set:
                    pos_set.append(word)
                if word in bottom_set:
                    neg_set.append(word)
        #    print("This sentence has an unsure prediction")
            if len(pos_set) != 0:
        #        print("This sentence has some positive words")
                for word in pos_set:
                    if word not in stopwords:
                        print(word)
            if len(neg_set) != 0:
        #        print("This sentence has some negative words")
                for word in neg_set:
                    if word not in stopwords or word == 'not' or word == 'but':
                        print(word)
       #     print("The probability of it being negative is", scores[i][0])
       #     print("The probability of it being positive is", scores[i][1])
       # print("---------------------------------------------------------")
    return top_set, bottom_set
    




def write_gold_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the truth.

    You will not be able to run this code, since the tsvfile is not
    accessible to you (it is the test labels).
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write(label)
            f.write("\n")
    f.close()
def semi_supervised_learning(unlabeled,sentiment,f,iters):
    import classify
    import numpy as np
    from sklearn.utils import shuffle
    import matplotlib.pyplot as plt
    cls = classify.train_classifier(sentiment.trainX,sentiment.trainy) # initial train with 0 unlabelled predicted
    initial_preds = cls.predict(unlabeled.X)
    factor = f # roughly about 10% of the corpus

  #  print(type(sentiment.trainX))
  #  print(type(sentiment.trainy))
    unlabeled.data_temp = unlabeled.data
   
    for i in range(iters):


        end_index = min(len(unlabeled.data),(i*factor) + factor)
        partition = unlabeled.data_temp[i*factor:end_index] # create partition of data
        #partition_matrix = sentiment.tfidf_vect.transform(partition) # create tfidf features on corpus
        partition_matrix = unlabeled.X[i*factor:end_index]
        yp = cls.predict(partition_matrix) # predict on this partition of unseen data to create labels
        decisions = cls.decision_function(partition_matrix)
        # predict on unseen portion of data
        #for j in range(len(decisions)):
        #    print(decisions[j])
        #print(decisions)
        #print(decisions)
        # append this data to the train to create new train with labels
        for j in range(len(partition)):
            # check the confidence on each prediction before appending
            if(abs(decisions[j]) > 3.5):
                #print("HI")
            # print(partition[j])
            # print(yp[j])
                sentiment.train_data.append(partition[j])

                sentiment.trainy = np.append(sentiment.trainy,yp[j])
        print(len(sentiment.train_data))
        print(sentiment.trainy.shape)
        sentiment.trainX = sentiment.tfidf_vect.transform(sentiment.train_data) # transform new training data with partition addition
        cls = classify.train_classifier(sentiment.trainX,sentiment.trainy) # train a new classifier
        classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev') # evaluate on dev portion

    return cls # return this new classifier

def lexicon_stuff():
    from nltk.corpus import opinion_lexicon
    print(opinion_lexicon.positive())
    print("-------------------------")
    print(opinion_lexicon.negative())

        


def write_basic_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the naive baseline.

    This baseline predicts POSITIVE for all the instances.
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write("POSITIVE")
            f.write("\n")
    f.close()

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("Please enter two arguments")
        sys.exit(1)
    if(sys.argv[1] == "run_model"):
        print("Reading data")
        tarfname = "data/sentiment.tar.gz"
        sentiment = read_files(tarfname)
        print("\nTraining classifier")
        import classify
        cls = classify.train_classifier(sentiment.trainX, sentiment.trainy)
        print("\nEvaluating")
        classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
        classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')

        print("\nReading unlabeled data")
        unlabeled = read_unlabeled(tarfname, sentiment)
        print(lexicon_stuff)
        cls = semi_supervised_learning(unlabeled, sentiment)
        print("Writing predictions to a file")
        write_pred_kaggle_file(unlabeled, cls, "data/sentiment-pred.csv", sentiment)
        #write_basic_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-basic.csv")

        # You can't run this since you do not have the true labels
        # print "Writing gold file"
        # write_gold_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-gold.csv")
    if(sys.argv[1] == "final"):
        print("Reading data")
        tarfname = "data/sentiment.tar.gz"
        sentiment = read_files(tarfname)
        print("\nTraining classifier")
        import classify
        cls = classify.train_classifier(sentiment.trainX, sentiment.trainy)
        print("\nEvaluating")
        classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
        classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')

        print("\nReading unlabeled data")
        print("in first task")
        unlabeled = read_unlabeled(tarfname, sentiment)
        print(lexicon_stuff)
        cls = semi_supervised_learning(unlabeled, sentiment,8000,12)
        first_classification_task(unlabeled, cls, sentiment)
    if(sys.argv[1] == "final2"):
        print("Reading data")
        tarfname = "data/sentiment2.tar.gz"
        sentiment = read_files(tarfname)
        print("\nTraining classifier")
        import classify
        cls = classify.train_classifier(sentiment.trainX, sentiment.trainy)
        print("\nEvaluating")
        classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
        classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')

        print("\nReading unlabeled data")
        print("in first task")
        unlabeled = read_unlabeled(tarfname, sentiment)
        print(lexicon_stuff)
        cls = semi_supervised_learning(unlabeled, sentiment,100,6)
        first_classification_task(unlabeled, cls, sentiment)

    if(sys.argv[1] == "graph"):
        import classify
        import numpy as np
        from sklearn.utils import shuffle
        import matplotlib.pyplot as plt
        tarfname = "data/sentiment.tar.gz"
        sentiment = read_files(tarfname)
        unlabeled = read_unlabeled(tarfname, sentiment)
        cls = classify.train_classifier(sentiment.trainX,sentiment.trainy) # initial train with 0 unlabelled predicted
        classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
        factor = 8000 # roughly about 10% of the corpus

    #  print(type(sentiment.trainX))
    #  print(type(sentiment.trainy))
        unlabeled.data_temp = unlabeled.data
        iterations = []
        validation_acc = []
        for i in range(12):
            iterations.append(i+1)
            end_index = min(len(unlabeled.data),(i*factor) + factor)
            partition = unlabeled.data_temp[i*factor:end_index] # create partition of data
            #partition_matrix = sentiment.tfidf_vect.transform(partition) # create tfidf features on corpus
            partition_matrix = unlabeled.X[i*factor:end_index]
            yp = cls.predict(partition_matrix) # predict on this partition of unseen data to create labels
            decisions = cls.decision_function(partition_matrix)
            # predict on unseen portion of data
            #for j in range(len(decisions)):
            #    print(decisions[j])
            #print(decisions)
            #print(decisions)
            # append this data to the train to create new train with labels
            for j in range(len(partition)):
                # check the confidence on each prediction before appending
                if(abs(decisions[j]) > 4.0):
                    #print("HI")
                # print(partition[j])
                # print(yp[j])
                    sentiment.train_data.append(partition[j])

                    sentiment.trainy = np.append(sentiment.trainy,yp[j])
            print(len(sentiment.train_data))
            print(sentiment.trainy.shape)
            sentiment.trainX = sentiment.tfidf_vect.transform(sentiment.train_data) # transform new training data with partition addition
            cls = classify.train_classifier(sentiment.trainX,sentiment.trainy) # train a new classifier
            ret = classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev') # evaluate on dev portion 
            validation_acc.append(ret)
        plt.plot(iterations,validation_acc)
        plt.title("dev accuracy as number of partitions of size 8000 increased and decision threshold of 3.0")
        plt.xlabel("partitions")
        plt.ylabel("accuracy")
        plt.show()

