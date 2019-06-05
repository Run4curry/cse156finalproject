from flask import Flask, request, render_template
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
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
top_set_one, bottom_set_one, stopwords, top_k_map_one, bottom_k_map_one, coff_map_one = server_sentiment.first_classification_task(unlabeled, cls, sentiment_one)

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
top_set_two, bottom_set_two, stopwords, top_k_map_two, bottom_k_map_two, coff_map_two = server_sentiment.first_classification_task(unlabeled, cls_spam, sentiment_two)



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
    top_k_coeff = []
    bottom_k_coeff = []
    sentence_coeff = []
    # top_k_words : [string]
    # bottom_k_words : [string]
    # sentence : string
    # probabilities: [floats]
    # positive words : [string]
    # negative words: [string]
    # prediction_type: POSITIVE, NEGATIVE, UNSURE

    coff_map = dict()
    coff_sum = 0.0
    for i in range(len(top_set_one)):
        top_k_coeff.append(coff_map_one[top_set_one[i]])
    for i in range(len(bottom_set_one)):
        bottom_k_coeff.append(coff_map_one[bottom_set_one[i]])
    
    for word in text.split():
        if word not in coff_map_one:
            coff_sum += 0.0
            sentence_coeff.append(0.0)
        else:
            coff_sum += abs(coff_map_one[word])
            sentence_coeff.append(coff_map_one[word])

    for word in text.split():
        if word not in coff_map_one:
            value = 0.0
        else:
            value = coff_map_one[word] / coff_sum
        coff_map[word] = value

    pos_set = []
    neg_set = []
    prediction_type = None

    for i in range(len(labels)):
        if labels[i] == "POSITIVE" and scores[i][1] >= 0.70:
            for word in text.split():
                if word in top_set_one and word not in stopwords:
                    pos_set.append(word)

            prediction_type = 'POSITIVE'

        elif labels[i] == "NEGATIVE" and scores[i][0] >= 0.70:
            for word in text.split():
                if word in bottom_set_one:
                    if word not in stopwords or word == 'not' or word == 'but':
                        neg_set.append(word)

            prediction_type = 'NEGATIVE'

        else:
            for word in text.split():
                if word in top_set_one and word not in stopwords:
                    pos_set.append(word)
                if word in bottom_set_one:
                    if word not in stopwords or word == 'not' or word == 'but':
                        neg_set.append(word)
            
            if(scores[i][0] > scores[i][1]):
                prediction_type = 'NEGATIVE NOT CONFIDENT'
            else:
                prediction_type = 'POSITIVE NOT CONFIDENT'
    print(pos_set)
    print(neg_set)
    print(top_k_coeff)
    print(bottom_k_coeff)
    print(sentence_coeff)

    return render_template('model.html', type='FOOD', sentence=text.split(), top_k_words=top_set_one, bottom_k_words=bottom_set_one, probabilities=scores.tolist(), positive_words=pos_set, negative_words=neg_set, prediction_type=prediction_type, weight=coff_map, top_k_coeff=top_k_coeff, bottom_k_coeff=bottom_k_coeff, sentence_coeff=sentence_coeff)

@app.route("/method2",methods=['POST'])
def method2():
    text = request.form['text']
    li = []
    li.append(text)
    print(type(sentiment_two))
    print(type(sentiment_two.tfidf_vect))
    data_point = sentiment_two.tfidf_vect.transform(li)
    yp = cls.predict(data_point)
    scores = cls.predict_proba(data_point)
    labels = sentiment_two.le.inverse_transform(yp)
    print(labels)
    print(scores)
    # top_k_words : [string]
    # bottom_k_words : [string]
    # sentence : string
    # probabilities: [floats]
    # positive words : [string]
    # negative words: [string]
    # prediction_type: POSITIVE, NEGATIVE, UNSURE
    top_k_coeff = []
    bottom_k_coeff = []
    sentence_coeff = []
    coff_map = dict()
    coff_sum = 0.0
    for i in range(len(top_set_one)):
        top_k_coeff.append(coff_map_two[top_set_two[i]])
    for i in range(len(bottom_set_one)):
        bottom_k_coeff.append(coff_map_two[bottom_set_two[i]])
    for word in text.split():
        if word not in coff_map_two:
            coff_sum += 0.0
            sentence_coeff.append(0.0)
        else:
            coff_sum += abs(coff_map_two[word])
            sentence_coeff.append(coff_map_two[word])

    for word in text.split():
        if word not in coff_map_two:
            value = 0.0
        else:
            value = coff_map_two[word] / coff_sum
        coff_map[word] = value

    pos_set = []
    neg_set = []
    prediction_type = None

    for i in range(len(labels)):
        if labels[i] == "POSITIVE" and scores[i][1] >= 0.70:
            for word in text.split():
                if word in top_set_two and word not in stopwords:
                    pos_set.append(word)
            prediction_type = 'POSITIVE'

        elif labels[i] == "NEGATIVE" and scores[i][0] >= 0.70:
            for word in text.split():
                if word in bottom_set_two:
                    if word not in stopwords or word == 'not' or word == 'but':
                        neg_set.append(word)
            prediction_type = 'NEGATIVE'

        else:
            for word in text.split():
                if word in top_set_two and word not in stopwords:
                    pos_set.append(word)
                if word in bottom_set_two:
                    if word not in stopwords or word == 'not' or word == 'but':
                        neg_set.append(word)
            
            if(scores[i][0] > scores[i][1]):
                prediction_type = 'NEGATIVE NOT CONFIDENT'
            else:
                prediction_type = 'POSITIVE NOT CONFIDENT'
    
    print(pos_set)
    print(neg_set)
    

    return render_template('model.html', type='SPAM', sentence=text.split(), top_k_words=top_set_two, bottom_k_words=bottom_set_two, probabilities=scores.tolist(), positive_words=pos_set, negative_words=neg_set, prediction_type=prediction_type, weight=coff_map, top_k_coeff=top_k_coeff, bottom_k_coeff=bottom_k_coeff, sentence_coeff=sentence_coeff)
