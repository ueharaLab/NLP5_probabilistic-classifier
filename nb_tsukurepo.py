import numpy as np
import pandas as pd
from naive_bayes_classifier import naive_bayes

tsukurepo_bow= pd.read_csv('./data/tsukurepo_bow.csv', encoding='ms932', sep=',',skiprows=0)

tsukurepo_bow=pd.concat([tsukurepo_bow.iloc[:,0],tsukurepo_bow.iloc[:,4:]],axis=1)

idx = np.arange(len(tsukurepo_bow))
print(idx)
np.random.shuffle(idx)
train_idx = int(len(tsukurepo_bow)*0.75)
index_train = idx[:train_idx]
index_test = idx[train_idx:]
train_bow = tsukurepo_bow.iloc[index_train,:]
test_bow = tsukurepo_bow.iloc[index_test,:]


nb = naive_bayes()
nb.fit(train_bow)
accuracy = nb.score(test_bow)

print('predict accuracy :{0} no.of test data :{1}'.format(accuracy,len(test_bow)))


