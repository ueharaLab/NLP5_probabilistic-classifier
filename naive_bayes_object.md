# ナイーブベイズのオブジェクト
学習と識別が独立しているとすると、k-nearest neighbor と同様に、knn.fit, knn.predict, のようにそれぞれ関数化（オブジェクト化）できる。  
1.[nb_tsukurepo.py](nb_tsukurepo.py)はツクレポのナイーブベイズによる学習・識別をオブジェクト指向で記述したもの。  
2. 内容は同じだが、fit, scoreのようにknnでの学習・識別と同じコーディングスタイルになっている。  
3. knnではfit,scoreはブラックボックスだったが、今回はそれらの中身も記述している [naive_bayes_classifier.py](naive_bayes_classifier.py)  

### 問題
1. [nb_tsukurepo.py](nb_tsukurepo.py)，[naive_bayes_classifiler.py](naive_bayes_classifier.py)と、[naive_bayes_tsukurepo.py](naive_bayes_tsukurepo.py)を比較して同じ機能を実現するのにどのような違いがあるのかを考えてみよ。
2. [nb_tsukurepo.py](nb_tsukurepo.py)の代わりに、学習と識別をそれぞれ関数でコーディングするとすると、学習済データは識別時にどのように参照することになるだろうか。
