# Doc2Vecをいい感じに評価したい

### Blog link
下記の構成を参考にnotebookを見れば理解できるようにしているが、  
リンク先の[blog](https://recruit.gmo.jp/engineer/jisedai/blog/doc2vec-evaluation/)では、このrepositoryの内容をまとめて説明している。

### 対象読者
「Doc2Vecやってみた」から次の一歩を踏み出す人達に役立つ内容かなぁ、と思ってます。

### Repositoryの内容
- Doc2Vecは教師なし学習なので、学習評価が難しい。このRepositoryで学習結果の評価方法を考える。
- 上記の評価方法を使って、異なるセットアップ（前処理やハイパーパラメータ）で学習させたDoc2Vecモデルを評価する。

##### 評価方法は大きく分けて以下の２つ
- 定性的評価: カテゴリを使わない場合（教師なしデータとして扱う）について、定性的に評価を行う。
- 定量的評価: カテゴリのある文章で、Doc2Vecのベクトルを教師あり学習に使うことで定量的な評価を行う。

2について補足すると、Doc2Vecを学習するのには、カテゴリ（教師データ）は不要。  
カテゴリは、学習した後のベクトルを定量的に評価するために使う。  

### Notebookの構成 (experiments/以下にある)
#### A. Deafult setup (preprocess_1)
1. データの説明＆データ加工（前処理1）: check_preproc1.ipynb
1. トレーニング: train_doc2vec_1-default.ipynb
1. 定性的評価: evaluation_wo_category_1-default
1. 定量的評価: evaluation_w_category_1-default

#### B. Hyper parameter tuning (preprocess_1)
1. ハイパーパラメータの説明: exp_params_setup1.ipynb
1. チューニング結果: exp_params_setup1.ipynb
1. 定量的評価: evaluation_w_category_1-opt.ipynb

#### C. Different preprocess (preprocess_2)
1. データ＆データ加工（前処理2）: check_preproc2.ipynb
1. チューニング結果: exp_params_setup2.ipynb
1. 定量的評価: evaluation_w_category_2-opt.ipynb
