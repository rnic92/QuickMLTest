import numpy as np
import pandas as pd
from collections import Counter


def getdata():
    datafile = pd.read_csv("SPAM text message 20170820 - Data.csv")
    # trainlabel = datafile
    # print(datafile.values)
    total = datafile.to_numpy()
    labels = total[:, 0]
    data = total[:, 1]
    # shuffle data for train and test
    a = np.vstack((data, labels)).T
    rng = np.random.default_rng()
    rng.shuffle(a)
    data = a[:, 0]
    labels = a[:, 1]
    return np.transpose(data), np.transpose(labels)


def getdata2():
    datafile = pd.read_csv("spam_or_not_spam.csv")
    total = datafile.to_numpy()
    data = total[:, 0]
    labels = total[:, 1]
    for i in range(len(labels)):
        if labels[i] == 1:
            labels[i] = "spam"
        else:
            labels[i] = "ham"
    a = np.vstack((data, labels)).T
    rng = np.random.default_rng()
    rng.shuffle(a)
    data = a[:, 0]
    labels = a[:, 1]
    return np.transpose(data), np.transpose(labels)


def combinedata(data1, data2, labels1, labels2):
    return np.append(data1, data2), np.append(labels1, labels2)


def splitdata(data, labels):
    """
    splits the data into training and testing sets
    """
    hamtotal = 0
    spamtotal = 0
    for i in range(len(data)):
        if(labels[i] == "ham"):
            hamtotal += 1
        if(labels[i] == "spam"):
            spamtotal += 1
    h = np.zeros(hamtotal)
    s = np.zeros(spamtotal)
    x = 0
    y = 0
    traind = []
    testd = []
    trainl = []
    testl = []
    for i in range(len(data)):
        if(labels[i] == "ham"):
            h[x] = i
            x += 1
        if(labels[i] == "spam"):
            s[y] = i
            y += 1
    for i in range(len(h)):
        if(i % 3 == 0):
            testd.append(data[int(h[i])])
            testl.append(labels[int(h[i])])
        else:
            traind.append(data[int(h[i])])
            trainl.append(labels[int(h[i])])
    for i in range(len(s)):
        if(i % 3 == 0):
            testd.append(data[int(s[i])])
            testl.append(labels[int(s[i])])
        else:
            traind.append(data[int(s[i])])
            trainl.append(labels[int(s[i])])

    trainl = np.array(trainl)
    traind = np.array(traind)
    testl = np.array(testl)
    testd = np.array(testd)
    # shuffle the data
    a = np.vstack((traind, trainl)).T
    rng = np.random.default_rng()
    rng.shuffle(a)
    traind = a[:, 0]
    trainl = a[:, 1]
    a = np.vstack((testd, testl)).T
    rng.shuffle(a)
    testd = a[:, 0]
    testl = a[:, 1]
    return traind, trainl, testd, testl


def featurextract(data, label):
    """
    Feature vectors returned for training and testing set
    features are length, >100, >150, contains word: free or Free, #, !, ...
    , other symbols,

    long: 22.5% of ham, 88% of spam
    xlong: 8.45% ham, 47% spam
    free: 1.2% ham, 14.25% spam
    #:  4.2% ham, 1.6% spam
    !: 11.567% ham, 51% spam
    elipse: 16.14% ham, 0.8% spam
    symbol: 9.4% ham, 26% spam
    winner: 0% ham, 2% spam

        commented out portions were for feature selection
        Chose features that seemed intuitive and differentiated
        Training data sufficiently well.  As shown above.
    """
    features = np.zeros((len(data), 12))
    # longham, xham = 0, 0
    # longspam, xspam = 0, 0
    # fspam, fham, tspam, tham = 0, 0, 0, 0
    # dspam, dham = 0, 0
    # pspam, pham = 0, 0
    # ham, spam = 0, 0
    # xham, xspam = 0, 0
    long = 100
    extralong = 150
    # espam, eham = 0, 0
    # cham, cspam = 0, 0

    for x in range(len(data)):
        data[x] = data[x].lower()
        # features[x][0] = len(data[x])
        # if(label[x] == "ham"):
        #     ham += 1
        # if(label[x] == "spam"):
        #     spam += 1
        if("free" in data[x] and label[x] == "spam"):
            # if(label[x] == "spam"):
            features[x][3] = 1
            # fspam += 1
        if(("win" in data[x] or "winner" in data[x]) and label[x] == "spam"):
            # if(label[x] == "spam"):
            features[8] = 1
            # cspam += 1
        if('$' in data[x] or '#' in data[x] or '-' in data[x]
                or '@' in data[x] and label[x] == "spam"):
            # if(label[x] == "spam"):
            features[x][7] = 1
            # dspam += 1
        if('%' in data[x] and label[x] == "spam"):
            # if(label[x] == "spam"):
            features[x][4] = 1
            # pspam += 1 # HERE
        if('...' in data[x] and label[x] == "ham"):
            # if(label[x] == "spam"):
            features[x][6] = 1
            # tspam += 1
        if(("txt" in data[x] or "text" in data[x]) and label[x] == "spam"):
            features[x][10] = 1
        if("hyperlink" in data[x] and label[x] == "spam"):
            features[x][11] = 1
        if('!' in data[x] and label[x] == "spam"):
            # if(label[x] == "spam"):
            features[x][5] = 1
            # espam += 1
        if(("cash" in data[x] or "money" in data[x]) and label[x] == "spam"):
            features[x][9] = 1
        if("urgent" in data[x] and label[x] == "spam"):
            features[x][0] = 1
        if len(data[x]) > long and label[x] == "spam":
            features[x][1] = 1
            # longspam += 1
            if(len(data[x]) > extralong):
                features[x][2] = 1
                # xspam += 1
    # print("long percents: ", longham/ham * 100, longspam/spam * 100)
    # print("extralong: ", xham/ham * 100, xspam/spam * 100)
    # print("free percents: ", fham/ham * 100, fspam/spam*100)
    # print("dollar percents: ", dham/ham * 100, dspam/spam*100)
    # print("pound percents: ", pham/ham * 100, pspam/spam*100)
    # print("elipse percents: ", tham/ham * 100, tspam/spam*100)
    # print("! percents: ", eham/ham * 100, espam/spam*100)
    # print("winner percents: ", cham/ham * 100, cspam/spam*100)
    return features


def bagofwords(data, label):
    hamdict = {}
    spamdict = {}
    for x in range(len(data)):
        data[x] = str(data[x])
        data[x] = data[x].lower()
        words = data[x].split()
        for word in words:
            if label[x] == "spam":
                if word in spamdict:
                    spamdict[word] += 1
                else:
                    spamdict[word] = 1
            else:
                if word in hamdict:
                    hamdict[word] += 1
                else:
                    hamdict[word] = 1
    h = Counter(hamdict)
    s = Counter(spamdict)
    ha = h.most_common(50)
    sa = s.most_common(50)
    print("ham")
    for i in ha:
        print(i[0], " :", i[1]/len(hamdict)*100, " ")
    print("\n\nSpam")
    for i in sa:
        print(i[0], " :", i[1]/len(spamdict)*100, " ")


def features(data, bag):
    """
    Featurearry will be a matrix of length of data input
    width, number of words in the bag.  Only used in association with
    bagofwords
    """

    featurearry = np.zeros((len(data), len(bag)))
    for x in range(len(data)):
        data[x] = data[x].lower()
        words = data[x].split()
        for word in range(len(bag)):
            if bag[word] in words:
                featurearry[x, word] = 1
    print(featurearry.shape)
    return featurearry


def labelfix(labels):
    fixed = np.zeros(len(labels))
    for x in range(len(labels)):
        if labels[x] == "spam":
            fixed[x] = 1
    return fixed
