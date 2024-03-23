import numpy as np
import pandas as pd


class naive_bayes:
    #def __init__(self, ):
        
    def fit(self,train_bow):
        # ------------- 事前確率の計算 ------------------------
        self.label = np.unique(train_bow['keyword'])
        num_train = len(train_bow)
        self.priors={l:0 for l in self.label}
        for l,val in self.priors.items():

            num = len(train_bow[train_bow['keyword']==l])
            self.priors[l]=num/num_train

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

        self.words_logProbs_df = pd.DataFrame(words_logProbs,index=labels, columns = tsukurepo_byLabel.columns.tolist() )

    def score(self, test_bow):
                
        # -----------------  識別 ----------------------------------------------
        def posterior_inference(words):
            
            posteriors = {l:0 for l in self.label}
            for l in self.label:
                log_likelihood = sum([self.words_logProbs_df.loc[l,w]*c for w,c in words.items()])
                #print(log_likelihood,priors[label])
                posterior = log_likelihood + self.priors[l]
                posteriors[l]=posterior
            
            max_label = max(posteriors, key=posteriors.get)
            return max_label



        true_counter=0
        for i,row in test_bow.iterrows():
            label = row['keyword']
            words = row.iloc[1:]
            label_infer = posterior_inference(words)
            print('predict:{0}  true value:{1}'.format(label_infer,label) )
            if label == label_infer:
                true_counter+=1

        accuracy = true_counter/len(test_bow)
        return accuracy