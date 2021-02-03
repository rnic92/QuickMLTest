# QuickMLTest
Spam filter ML training algorithm

Nicolas rohr
20 Dec 2020
Version 2.2

Quick training algorithm without using pyTorch, sci-kit, or TF.
Created on python 3.7. Uses Pandas

Single pass, single layer laplace approximation.

simply run !python run.py

Training and Testing data in CSV files included.  Can import your own if you choose.  Defined in Startdata.py.  Test data is randomly selected from entire dataset.  Sets are shuffled.

Error rate for training data/testing data included usually falls ~0.6%
