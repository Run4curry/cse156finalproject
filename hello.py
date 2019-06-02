from flask import Flask, request, render_template
app = Flask(__name__)
import sys
import server_sentiment


print("Reading data")
tarfname = "data/sentiment.tar.gz"
sentiment_one = server_sentiment.read_files(tarfname)
print("\nTraining classifier")
import classify
cls = classify.train_classifier(sentiment_one.trainX, sentiment_one.trainy)
print("\nEvaluating")
classify.evaluate(sentiment_one.trainX, sentiment_one.trainy, cls, 'train')
classify.evaluate(sentiment_one.devX, sentiment_one.devy, cls, 'dev')

print("\nReading unlabeled data")
print("in first task")
unlabeled = server_sentiment.read_unlabeled(tarfname, sentiment_one)
cls = server_sentiment.semi_supervised_learning(unlabeled, sentiment_one,8000,12)
top_set_one, bottom_set_one, stopwords = server_sentiment.first_classification_task(unlabeled, cls, sentiment_one)
    
print("Reading data")
tarfname = "data/sentiment2.tar.gz"
sentiment_two = server_sentiment.read_files(tarfname)
print("\nTraining classifier")
import classify
cls_spam = classify.train_classifier(sentiment_two.trainX, sentiment_two.trainy)
print("\nEvaluating")
classify.evaluate(sentiment_two.trainX, sentiment_two.trainy, cls_spam, 'train')
classify.evaluate(sentiment_two.devX, sentiment_two.devy, cls_spam, 'dev')

print("\nReading unlabeled data")
print("in first task")
unlabeled = server_sentiment.read_unlabeled(tarfname, sentiment_two)
cls_spam = server_sentiment.semi_supervised_learning(unlabeled, sentiment_two,100,6)
top_set_two, bottom_set_two, stopwords = server_sentiment.first_classification_task(unlabeled, cls_spam, sentiment_two)

  

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/method1",methods=['POST'])
def method1():
    text = request.form['text']
    li = []
    li.append(text)
    print(type(sentiment_one))
    print(type(sentiment_one.tfidf_vect))
    data_point = sentiment_one.tfidf_vect.transform(li)
    yp = cls.predict(data_point)
    scores = cls.predict_proba(data_point)
    labels = sentiment_one.le.inverse_transform(yp)
    print(labels)
    print(scores)
    # top_k_words : [string]
    # bottom_k_words : [string]
    # sentence : string
    # probabilities: [floats]
    # positive words : [string]
    # negative words: [string]
    # prediction_type: POSITIVE, NEGATIVE, UNSURE

    for i in range(len(labels)):
        # confidence postive prediction
   #     print("---------------------------------------------------------")
    #    print(unlabeled.data[i])
        #print(scores[i])
        #print(labels[i])
        if labels[i] == "POSITIVE" and scores[i][1] >= 0.70:
            result = []
            for word in unlabeled.data[i].split():
                if word in top_set_one:
                    result.append(word)

    #        print("This sentence is positive because of these words")
            for word in result:
                if word not in stopwords:
                    print(word)
    #        print("The probability of it being positive is", scores[i][1])

        elif labels[i] == "NEGATIVE" and scores[i][0] >= 0.70:
            result = []
            for word in unlabeled.data[i].split():
                if word in bottom_set_one:
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
                if word in top_set_one:
                    pos_set.append(word)
                if word in bottom_set_one:
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
    return "SUCCESS BITCH"
