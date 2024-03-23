# ナイーブベイズをpythonで実装する

1. [naive_bayes.py](naive_bayes.py)を修正して、手計算と同じ結果が得られるようにせよ。修正箇所は### ###で示している。   
2. 学習済のデータ,すなわち事前確率と尤度はコーディング上でどの変数に記憶されるか？
3. 学習が終わると、任意のテストデータを識別できることに注意。つまり、学習と識別はコーディング上独立していることを理解すること。 

``` python
tsukurepo_bow= pd.read_csv('syu_prin.csv', encoding='ms932', sep=',',skiprows=0)


# ------------- 事前確率の計算 ------------------------
num_train = len(tsukurepo_bow)
priors={'シュークリーム':0,'プリン':0} #事前確率を初期化
for l,val in priors.items():

    num = len(tsukurepo_bow[tsukurepo_bow['label']==l]) # これは何をやっているだろうか？
    priors[l]= ###   ###

# ----------- 尤度 -----------------------------------
tsukurepo_byLabel = tsukurepo_bow.groupby('label').sum() # label（シュー/プリン）毎の語彙小計を計算
total_val = np.sum(tsukurepo_bow.values[:,1:])
words_logProbs=[]
labels = []
for label, words in tsukurepo_byLabel.iterrows():
    # label,wordsにはそれぞれ何が入っているか？
    count_total = np.sum(words.values)
    # 加算スムージングを行った対数尤度
    words_prob = np.log((###   ###)/(###   ###))
    words_logProbs.append(words_prob)
    labels.append(label)
# index:ラベル(シュー/プリン)　columns：語彙であるようなdataframeを作成。値には上記の加算スムージング済の尤度が入る。このコーディングでなぜそのようなdataframeを作れるのか？（csvに書き出しているのでdataframeのイメージは確認できる）
words_logProbs_df = pd.DataFrame(words_logProbs,index=labels, columns = tsukurepo_byLabel.columns.tolist() )
with codecs.open("./data/syu_prin_logLikelihood.csv", "w", "ms932", "ignore") as f: 
    
    words_logProbs_df.to_csv(f, index=True, encoding="ms932", mode='w', header=True)

# -----------------  識別 ----------------------------------------------
def posterior_inference(words):
    label=['シュークリーム','プリン']
    posteriors = {'シュークリーム':0,'プリン':0}
    for l in label:
        # 以下のコーディングで尤度関数の和を計算している。なぜこれでOKか？  
        # posterior[l]には何が入るか？スライドp.20 の識別に示した式とコーディングの変数との対応関係を述べよ
        log_likelihood = sum([words_logProbs_df.loc[l,w]*c for w,c in words.items()])
        
        posterior = log_likelihood + priors[l]
        
        posteriors[l]=posterior
       
    max_label = max(posteriors, key=posteriors.get)
    return max_label,posteriors

test1 = {'パイ':2, '生クリーム':1, '砂糖':2} 
test2 = {'生クリーム':3,'卵':1,'砂糖':3} 
label_infer,prob = posterior_inference(test1)
print('predict:{0}  prob:{1}'.format(label_infer,prob) )
label_infer,prob = posterior_inference(test2)
print('predict:{0}  prob:{1}'.format(label_infer,prob) )
```