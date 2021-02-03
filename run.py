import Startdata
import Spamfilter
# import tracemalloc
# import time


def test(theta, features, labels):
    """run the testing data using theta as weights"""
    N = labels.size  # sample size 12665
    correct = 0
    for i in range(N):
        temp = Spamfilter.sigmoid(Spamfilter.activation(theta, features[i]))
        if (temp > 0.5 and labels[i] == 1):
            correct += 1
        elif(temp < 0.5 and labels[i] == 0):
            correct += 1
    print(correct, "correct out of", labels.size)
    print("testing error rate: ", 100-(correct/N*100))


def run():
    """
    Main driving function.  Retrieves the data from file given in Startdata.py
    Splits into 2/3 training, 1/3 testing.  Extracts features specified in
    Startdata.py.  Trains weight in Spamfilter.py.  Uses Laplace approximation.
    Line search gradient descent.  Bayesian Logistic Regression
    """
    # all data and labels
    # tracemalloc.start()
    # start = time.time()
    data, labels = Startdata.getdata()  # texts
    data2, labels2 = Startdata.getdata2()  # emails
    # Startdata.bagofwords(data2, labels2)
    data, labels = Startdata.combinedata(data, data2, labels, labels2)
    # split into training and testing. 1/3 test, 2/3 train
    traind, trainl, testd, testl = Startdata.splitdata(data, labels)

    # labels
    trainlabels = Startdata.labelfix(trainl)
    testlabels = Startdata.labelfix(testl)

    # selective features
    #
    # extract features for use. in the shape of NxD
    # N is number of samples, D is number of features
    # current, peak = tracemalloc.get_traced_memory()
    trainfeat = Startdata.featurextract(traind, trainl)
    testfeat = Startdata.featurextract(testd, testl)
    # theta is the weights in a D+1 X 1 array
    theta = Spamfilter.train(trainfeat, trainlabels)
    #
    # trying bag of words
    #

    # Startdata.featurextract(data, labels)
    # error rate was 1.69% for trainingdata
    #   2.21% for testing data
    # bag, tfeat = Startdata.bagofwords(traind)
    # theta = Spamfilter.train(tfeat, trainlabels)
    # testfeat = Startdata.features(testd, bag)

    test(theta, testfeat, testlabels)
    # tracemalloc.stop()
    # done = time.time()
    # print(f"Current memory usage is {current / 10**6} MB; Peak was {peak / 10**6} MB")
    # print("time to complete", done - start)
    # NTR 12/1/2020 current best featextraction at 25 iterations is about
    # 0.7-1% error for
    # trainingdata and testing data
    # NTR 12/2/2020 bag of words  at 25 iterations
    # 1.69% training error, 2.21% testing error
    # NTR 12/2/2020 bag of words, 25 iter, removal of some features
    # NTR 12/3/2020 featextraction 20 iterations, new features, emails inc
    # 0.59% error on training. 0.63% testing error


run()  # driver
