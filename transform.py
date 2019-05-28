def transform():
    with open('spam_data.csv', 'r') as file1:
        with open('train_spam_data.tsv', 'w') as train_file:
            with open('dev_spam_data.tsv', 'w') as dev_file:
                with open('unlabeled_spam_data.tsv', 'w') as test_file:
                    i = 0
                    for line in file1:
                        if i <= 4999:
                            line = line.replace(',', '\t', 1)
                            if line[0:3] == 'ham':
                                line = line.replace('ham', 'POSITIVE', 1)
                            else:
                                line = line.replace('spam', 'NEGATIVE', 1)
                        else:
                            if line[0:3] == 'ham':
                                line = line[4:]
                            else:
                                line = line[5:]
                        i += 1
                        if i <= 4000:
                            train_file.write(line)
                        elif i <= 5000 and i >= 4000:
                            dev_file.write(line)
                        else:
                            test_file.write(line)

transform()