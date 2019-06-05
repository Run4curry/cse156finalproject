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


print("Reading data")
tarfname = "data/sentiment3.tar.gz"
sentiment_three = server_sentiment.read_files(tarfname)
print("\nTraining classifier")
import classify
cls_gender = classify.train_classifier(sentiment_three.trainX, sentiment_three.trainy)
print("\nEvaluating")
classify.evaluate(sentiment_three.trainX, sentiment_three.trainy, cls_gender, 'train')
classify.evaluate(sentiment_three.devX, sentiment_three.devy, cls_gender, 'dev')

print("\nReading unlabeled data")
print("in first task")
unlabeled = server_sentiment.read_unlabeled(tarfname, sentiment_three)
cls_gender = server_sentiment.semi_supervised_learning(unlabeled, sentiment_three,100,19)
top_set_three, bottom_set_three, stopwords, top_k_map_three, bottom_k_map_three, coff_map_three = server_sentiment.first_classification_task(unlabeled, cls_gender, sentiment_three)



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

    coff_map = []
    coff_map2 = []
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
        coff_map.append(value)

    words = text.split()
    bigrams = []
    bigrams_coff = []
    for i in range(len(words) - 1):
        bigram = words[i] + ' ' + words[i + 1]
        bigrams.append(bigram)
        if bigram in coff_map_one:
            coff_sum += abs(coff_map_one[bigram])
            bigrams_coff.append(coff_map_one[bigram])
        else:
            bigrams_coff.append(0.0)

    trigrams = []
    trigrams_coff = []
    for i in range(len(words) - 2):
        trigram = words[i] + ' ' + words[i + 1] + ' ' + words[i + 2]
        trigrams.append(trigram)
        if trigram in coff_map_one:
            coff_sum += abs(coff_map_one[trigram])
            trigrams_coff.append(coff_map_one[trigram])
        else:
            trigrams_coff.append(0.0)


# ----------------------------------
    for i in range(len(words)):
        # print(words[i])
        value = 0.0
        if words[i] not in coff_map_one:
            value = 0.0
            # print(words[i]," not in coff map one")
        else:
            value += coff_map_one[words[i]]
            # print(words[i]," in coff map one")
            if i == 0: #first word case
                if len(words) > 1:
                    bigram = words[0] + ' ' + words[1]
                    if bigram in coff_map_one:
                        value += (0.5*coff_map_one[bigram])
                    if len(words) > 2:
                        trigram = words[0] + ' ' + words[1] + ' ' + words[2]
                        if trigram in coff_map_one:
                            value += ((1/3)*coff_map_one[trigram])
            elif i == 1: #second word case
                bigram1 = words[0] + ' ' + words[1]
                if bigram1 in coff_map_one:
                    value += (0.5*coff_map_one[bigram1])
                if len(words) > 2:
                    bigram2 = words[1] + ' ' + words[2]
                    if bigram2 in coff_map_one:
                        value += (0.5*coff_map_one[bigram2])
                    trigram1 = words[0] + ' ' + words[1] + ' ' + words[2]
                    if trigram1 in coff_map_one:
                        value += ((1/3)*coff_map_one[trigram1])
                    if len(words) > 3:
                        trigram2 = words[1] + ' ' + words[2] + ' ' + words[3]
                        if trigram2 in coff_map_one:
                            value += ((1/3)*coff_map_one[trigram2])
            elif i == len(words) - 2: #second to last word case
                bigram1 = words[i-1] + ' ' + words[i]
                if bigram1 in coff_map_one:
                    value += (0.5*coff_map_one[bigram1])
                bigram2 = words[i] + ' ' + words[i+1]
                if bigram2 in coff_map_one:
                    value += (0.5*coff_map_one[bigram2])
                trigram1 = words[i-2] + ' ' + words[i-1] + ' ' + words[i]
                if trigram1 in coff_map_one:
                    value += ((1/3)*coff_map_one[trigram1])
                trigram2 = words[i-1] + ' ' + words[i] + ' ' + words[i+1]
                if trigram2 in coff_map_one:
                    value += ((1/3)*coff_map_one[trigram2])
            elif i == len(words) - 1: #last word case
                bigram1 = words[i-1] + ' ' + words[i]
                if bigram1 in coff_map_one:
                    value += (0.5*coff_map_one[bigram1])
                trigram1 = words[i-2] + ' ' + words[i-1] + ' ' + words[i]
                if trigram1 in coff_map_one:
                    value += ((1/3)*coff_map_one[trigram1])
            else:
                bigram1 = words[i-1] + ' ' + words[i]
                if bigram1 in coff_map_one:
                    value += (0.5*coff_map_one[bigram1])
                trigram1 = words[i-2] + ' ' + words[i-1] + ' ' + words[i]
                if trigram1 in coff_map_one:
                    value += ((1/3)*coff_map_one[trigram1])
                if len(words) > i+1:
                    bigram2 = words[i] + ' ' + words[i+1]
                    if bigram2 in coff_map_one:
                        value += (0.5*coff_map_one[bigram2])
                    trigram2 = words[i-1] + ' ' + words[i] + ' ' + words[i+1]
                    if trigram2 in coff_map_one:
                        value += ((1/3)*coff_map_one[trigram2])
                    if len(words) > i+2:
                        trigram3 = words[i] + ' ' + words[i+1] + ' ' + words[i+2]
                        if trigram3 in coff_map_one:
                            value += ((1/3)*coff_map_one[trigram3])
            value /= coff_sum
        coff_map2.append(value)
    # print("coff map:",coff_map2)
# ----------------------------------

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

    return render_template('model.html', type='FOOD', sentence=text.split(), top_k_words=top_set_one, bottom_k_words=bottom_set_one, probabilities=scores.tolist(), positive_words=pos_set, negative_words=neg_set, prediction_type=prediction_type, weight=coff_map, weight2=coff_map2, top_k_coeff=top_k_coeff, bottom_k_coeff=bottom_k_coeff, sentence_coeff=sentence_coeff, bigrams=bigrams, bigrams_coff=bigrams_coff, trigrams=trigrams, trigrams_coff=trigrams_coff)

@app.route("/method2",methods=['POST'])
def method2():
    text = request.form['text']
    li = []
    li.append(text)
    print(type(sentiment_two))
    print(type(sentiment_two.tfidf_vect))
    data_point = sentiment_two.tfidf_vect.transform(li)
    yp = cls_spam.predict(data_point)
    scores = cls_spam.predict_proba(data_point)
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
    coff_map = []
    coff_map2 = []
    coff_sum = 0.0
    for i in range(len(top_set_two)):
        top_k_coeff.append(coff_map_two[top_set_two[i]])
    for i in range(len(bottom_set_two)):
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
        coff_map.append(value)

    words = text.split()
    bigrams = []
    bigrams_coff = []
    for i in range(len(words) - 1):
        bigram = words[i] + ' ' + words[i + 1]
        bigrams.append(bigram)
        if bigram in coff_map_two:
            bigrams_coff.append(coff_map_two[bigram])
        else:
            bigrams_coff.append(0.0)

    trigrams = []
    trigrams_coff = []
    for i in range(len(words) - 2):
        trigram = words[i] + ' ' + words[i + 1] + ' ' + words[i + 2]
        trigrams.append(trigram)
        if trigram in coff_map_two:
            trigrams_coff.append(coff_map_two[trigram])
        else:
            trigrams_coff.append(0.0)

# ----------------------------------
    for i in range(len(words)):
        # print(words[i])
        value = 0.0
        if words[i] not in coff_map_two:
            value = 0.0
            # print(words[i]," not in coff map one")
        else:
            value += coff_map_two[words[i]]
            # print(words[i]," in coff map one")
            if i == 0: #first word case
                if len(words) > 1:
                    bigram = words[0] + ' ' + words[1]
                    if bigram in coff_map_two:
                        value += (0.5*coff_map_two[bigram])
                    if len(words) > 2:
                        trigram = words[0] + ' ' + words[1] + ' ' + words[2]
                        if trigram in coff_map_two:
                            value += ((1/3)*coff_map_two[trigram])
            elif i == 1: #second word case
                bigram1 = words[0] + ' ' + words[1]
                if bigram1 in coff_map_two:
                    value += (0.5*coff_map_two[bigram1])
                if len(words) > 2:
                    bigram2 = words[1] + ' ' + words[2]
                    if bigram2 in coff_map_two:
                        value += (0.5*coff_map_two[bigram2])
                    trigram1 = words[0] + ' ' + words[1] + ' ' + words[2]
                    if trigram1 in coff_map_two:
                        value += ((1/3)*coff_map_two[trigram1])
                    if len(words) > 3:
                        trigram2 = words[1] + ' ' + words[2] + ' ' + words[3]
                        if trigram2 in coff_map_two:
                            value += ((1/3)*coff_map_two[trigram2])
            elif i == len(words) - 2: #second to last word case
                bigram1 = words[i-1] + ' ' + words[i]
                if bigram1 in coff_map_two:
                    value += (0.5*coff_map_two[bigram1])
                bigram2 = words[i] + ' ' + words[i+1]
                if bigram2 in coff_map_two:
                    value += (0.5*coff_map_two[bigram2])
                trigram1 = words[i-2] + ' ' + words[i-1] + ' ' + words[i]
                if trigram1 in coff_map_two:
                    value += ((1/3)*coff_map_two[trigram1])
                trigram2 = words[i-1] + ' ' + words[i] + ' ' + words[i+1]
                if trigram2 in coff_map_two:
                    value += ((1/3)*coff_map_two[trigram2])
            elif i == len(words) - 1: #last word case
                bigram1 = words[i-1] + ' ' + words[i]
                if bigram1 in coff_map_two:
                    value += (0.5*coff_map_two[bigram1])
                trigram1 = words[i-2] + ' ' + words[i-1] + ' ' + words[i]
                if trigram1 in coff_map_two:
                    value += ((1/3)*coff_map_two[trigram1])
            else:
                bigram1 = words[i-1] + ' ' + words[i]
                if bigram1 in coff_map_two:
                    value += (0.5*coff_map_two[bigram1])
                trigram1 = words[i-2] + ' ' + words[i-1] + ' ' + words[i]
                if trigram1 in coff_map_two:
                    value += ((1/3)*coff_map_two[trigram1])
                if len(words) > i+1:
                    bigram2 = words[i] + ' ' + words[i+1]
                    if bigram2 in coff_map_two:
                        value += (0.5*coff_map_two[bigram2])
                    trigram2 = words[i-1] + ' ' + words[i] + ' ' + words[i+1]
                    if trigram2 in coff_map_two:
                        value += ((1/3)*coff_map_two[trigram2])
                    if len(words) > i+2:
                        trigram3 = words[i] + ' ' + words[i+1] + ' ' + words[i+2]
                        if trigram3 in coff_map_two:
                            value += ((1/3)*coff_map_two[trigram3])
            value /= coff_sum
        coff_map2.append(value)
    # print("coff map:",coff_map2)
# ----------------------------------


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


    return render_template('model.html', type='SPAM', sentence=text.split(), top_k_words=top_set_two, bottom_k_words=bottom_set_two, probabilities=scores.tolist(), positive_words=pos_set, negative_words=neg_set, prediction_type=prediction_type, weight=coff_map, weight2=coff_map2, top_k_coeff=top_k_coeff, bottom_k_coeff=bottom_k_coeff, sentence_coeff=sentence_coeff, bigrams=bigrams, bigrams_coff=bigrams_coff, trigrams=trigrams, trigrams_coff=trigrams_coff)


@app.route("/method3",methods=['POST'])
def method3():
    text = request.form['text']
    li = []
    li.append(text)
    print(type(sentiment_three))
    print(type(sentiment_three.tfidf_vect))
    data_point = sentiment_three.tfidf_vect.transform(li)
    yp = cls_gender.predict(data_point)
    scores = cls_gender.predict_proba(data_point)
    labels = sentiment_three.le.inverse_transform(yp)
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
    for i in range(len(top_set_three)):
        top_k_coeff.append(coff_map_three[top_set_three[i]])
    for i in range(len(bottom_set_three)):
        bottom_k_coeff.append(coff_map_three[bottom_set_three[i]])
    for word in text.split():
        if word not in coff_map_three:
            coff_sum += 0.0
            sentence_coeff.append(0.0)
        else:
            coff_sum += abs(coff_map_three[word])
            sentence_coeff.append(coff_map_three[word])

    # for word in text.split():
    #     if word not in coff_map_three:
    #         value = 0.0
    #     else:
    #         value = coff_map_three[word] / coff_sum
    #     coff_map[word] = value

    words = text.split()
    bigrams = []
    bigrams_coff = []
    for i in range(len(words) - 1):
        bigram = words[i] + ' ' + words[i + 1]
        bigrams.append(bigram)
        if bigram in coff_map_three:
            coff_sum += abs(coff_map_three[bigram])
            bigrams_coff.append(coff_map_three[bigram])
        else:
            bigrams_coff.append(0.0)

    trigrams = []
    trigrams_coff = []
    for i in range(len(words) - 2):
        trigram = words[i] + ' ' + words[i + 1] + ' ' + words[i + 2]
        trigrams.append(trigram)
        if trigram in coff_map_three:
            coff_sum += abs(coff_map_three[trigram])
            trigrams_coff.append(coff_map_three[trigram])
        else:
            trigrams_coff.append(0.0)

    for i in range(len(words)):
        if word not in coff_map_three:
            value = 0.0
        else:
            value += coff_map_three[words[i]]
            if i == 0: #first word case
                if len(words) > 1:
                    bigram = words[0] + ' ' + words[1]
                    value += (0.5*coff_map_three[bigram])
                    if len(words) > 2:
                        trigram = words[0] + ' ' + words[1] + ' ' + words[2]
                        value += ((1/3)*coff_map_three[trigram])
            elif i == 1: #second word case
                bigram1 = words[0] + ' ' + words[1]
                value += (0.5*coff_map_three[bigram1])
                if len(words) > 2:
                    bigram2 = words[1] + ' ' + words[2]
                    value += (0.5*coff_map_three[bigram2])
                    trigram1 = words[1] + ' ' + words[2] + ' ' + words[3]
                    value += ((1/3)*coff_map_three[trigram1])
                    if len(words) > 3:
                        trigram2 = words[2] + ' ' + words[3] + ' ' + words[4]
                        value += ((1/3)*coff_map_three[trigram2])
            elif i == len(words) - 2: #second to last word case
                bigram1 = words[i-1] + ' ' + words[i]
                value += (0.5*coff_map_three[bigram1])
                bigram2 = words[i] + ' ' + words[i+1]
                value += (0.5*coff_map_three[bigram2])
                trigram1 = words[i-2] + ' ' + words[i-1] + ' ' + words[i]
                value += ((1/3)*coff_map_three[trigram1])
                trigram2 = words[i-1] + ' ' + words[i] + ' ' + words[i+1]
                value += ((1/3)*coff_map_three[trigram2])
            elif i == len(words) - 1: #last word case
                bigram1 = words[i-1] + ' ' + words[i]
                value += (0.5*coff_map_three[bigram1])
                trigram1 = words[i-2] + ' ' + words[i-1] + ' ' + words[i]
                value += ((1/3)*coff_map_three[trigram1])
            else:
                bigram1 = words[i-1] + ' ' + words[i]
                value += (0.5*coff_map_three[bigram1])
                trigram1 = words[i-2] + ' ' + words[i-1] + ' ' + words[i]
                value += ((1/3)*coff_map_three[trigram1])
                if len(words) > i+1:
                    bigram2 = words[i] + ' ' + words[i+1]
                    value += (0.5*coff_map_three[bigram2])
                    trigram2 = words[i-1] + ' ' + words[i] + ' ' + words[i+1]
                    value += ((1/3)*coff_map_three[trigram2])
                    if len(words) > i+2:
                        trigram3 = words[i] + ' ' + words[i+1] + ' ' + words[i+2]
                        value += ((1/3)*coff_map_three[trigram3])
            value /= coff_sum
        coff_map[word] = value

    pos_set = []
    neg_set = []
    prediction_type = None

    for i in range(len(labels)):
        if labels[i] == "POSITIVE" and scores[i][1] >= 0.70:
            for word in text.split():
                if word in top_set_three and word not in stopwords:
                    pos_set.append(word)
            prediction_type = 'POSITIVE'

        elif labels[i] == "NEGATIVE" and scores[i][0] >= 0.70:
            for word in text.split():
                if word in bottom_set_three:
                    if word not in stopwords or word == 'not' or word == 'but':
                        neg_set.append(word)
            prediction_type = 'NEGATIVE'

        else:
            for word in text.split():
                if word in top_set_three and word not in stopwords:
                    pos_set.append(word)
                if word in bottom_set_three:
                    if word not in stopwords or word == 'not' or word == 'but':
                        neg_set.append(word)

            if(scores[i][0] > scores[i][1]):
                prediction_type = 'NEGATIVE NOT CONFIDENT'
            else:
                prediction_type = 'POSITIVE NOT CONFIDENT'

    print(pos_set)
    print(neg_set)


    return render_template('model.html', type='GENDER', sentence=text.split(), top_k_words=top_set_three, bottom_k_words=bottom_set_three, probabilities=scores.tolist(), positive_words=pos_set, negative_words=neg_set, prediction_type=prediction_type, weight=coff_map, top_k_coeff=top_k_coeff, bottom_k_coeff=bottom_k_coeff, sentence_coeff=sentence_coeff, bigrams=bigrams, bigrams_coff=bigrams_coff, trigrams=trigrams, trigrams_coff=trigrams_coff)
