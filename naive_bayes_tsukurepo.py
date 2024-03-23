
import numpy as np
import pandas as pd


tsukurepo_bow= pd.read_csv('./data/tsukurepo_bow.csv', encoding='ms932', sep=',',skiprows=0)

tsukurepo_bow=pd.concat([tsukurepo_bow.iloc[:,0],tsukurepo_bow.iloc[:,4:]],axis=1)

idx = np.arange(len(tsukurepo_bow))
print(idx)
np.random.shuffle(idx)
train_idx = int(len(tsukurepo_bow)*0.75)
index_train = idx[:train_idx]
index_test = idx[train_idx:]





# ------------- 事前確率の計算 ------------------------
train_bow = tsukurepo_bow.iloc[index_train,:]
num_train = len(train_bow)
priors={'杏仁豆腐':0,'シュークリーム':0,'プリン':0}
for l,val in priors.items():

    num = len(train_bow[train_bow['keyword']==l])
    priors[l]=num/num_train

# ----------- 尤度 -----------------------------------
tsukurepo_byLabel = train_bow.groupby('keyword').sum()
total_val = np.sum(train_bow.values[:,1:])
words_logProbs=[]
labels = []
for label, words in tsukurepo_byLabel.iterrows():

    count_total = np.sum(words.values)
    words_prob = np.log((words.values+1)/(count_total+total_val))
    words_logProbs.append(words_prob)
    labels.append(label)

words_logProbs_df = pd.DataFrame(words_logProbs,index=labels, columns = tsukurepo_byLabel.columns.tolist() )

# -----------------  識別 ----------------------------------------------
def posterior_inference(words):
    label=['杏仁豆腐','シュークリーム','プリン']
    posteriors = {'杏仁豆腐':0,'シュークリーム':0,'プリン':0}
    for l in label:
        log_likelihood = sum([words_logProbs_df.loc[l,w]*c for w,c in words.items()])
        #print(log_likelihood,priors[label])
        posterior = log_likelihood + priors[l]
        posteriors[l]=posterior
       
    max_label = max(posteriors, key=posteriors.get)
    return max_label

test_bow = tsukurepo_bow.iloc[index_test,:]

true_counter=0
for i,row in test_bow.iterrows():
    label = row['keyword']
    words = row.iloc[1:]
    label_infer = posterior_inference(words)
    print('predict:{0}  true value:{1}'.format(label_infer,label) )
    if label == label_infer:
        true_counter+=1

accuracy = true_counter/len(test_bow)

print('predict accuracy :{0} no.of test data :{1}'.format(accuracy,len(test_bow)))

