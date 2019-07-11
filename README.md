# Kaggle Earthquake Predict

__Competitions:__ [LANL Earthquake Prediction](https://www.kaggle.com/c/LANL-Earthquake-Prediction)  
__Rank:__ 186/4540 (Top 4%, silver)  
__Task:__ Predict earthquake happen time, regression problem  
__Data:__ trainset:length 629145480 signal record, testset:2624 signal record with a length of 150000  
__Note:__ This is a code backup, it's not runable due to the difference file path  



## Solution  
1. __采样__  
	* 如果训练集采用不重复采样，只能生成~4k个样本，样本量过小，容易overfit  
	* 如果采用有重复采样2w个，意味着每个样本都包含3个左右别的样本的片段，可能造成冗余，影响训练速度  
	* 大致试验了几次，选择有重复采样1w个样本  
2. __交叉验证：__  
	* 选择5fold交叉验证，减小过拟合风险  
	* 所有步骤，哪怕不同的stage，也要严格遵守同一份cv，避免泄露  
3. __提取DL Features__  
	* 将采样训练集reshap成(100,1500)和(25,6000)的片段，这里仍然需要提取片段的手工特征，而后放到LSTM和CNN网络里提取DL特征  
	* LSTM的效果比CNN好  
	* 可以考虑加self-attention，或者直接用类似bert的结构，效果应该能有提升  
	* 为了防止过拟合，每个模型提取的特征维度选择10（甚至可以更小）  
4. __提取统计特征__  
	* mean,std,min,max,分位数,等等一些基础特征  
	* tsfresh.feature_extraction的一些高级特征  
	* 原始波形经过fft，频带滤波，窗口平滑后的基础特征和高级特征  
	* 加dropout，weight decay，防止过拟合  
5. __特征生成和过滤__：refer feature_importance.py  
	* 将统计特征随机组合，得到新的特征，通过LGB筛选出重要性高的特征~2w(每次选择topn，重复若干次)  
	* 通过ks检验（scipy.stats.ks_2samp），检查特征在训练集和测试集上分布是否相同，筛除分布差异大的  
	* 通过相似度检测（pandas的corr），筛除互相关性高的特征，因为相关性高的特征会产生冗余  
	* 最终选择20个统计特征  
6. __多模型Predict__：将DL特征和统计特征拼接，得到片段特征向量，通过NN，RF，SVR，LGB，XGB等方法进行预测  
	* NN加dropout，weight decay， LGB、XGB、RF减小树深和叶子数，防止过拟合  
7. __模型融合__：这里尝试了两种方案  
	* 直接取平均  
	* stacking，第二阶段用regression求权重，最后结果加权平均  
	* 在public LB和CV上stacking得分高，实际priveate LB上直接取平均好，能到80名，这再次证明了这次比赛的训练集，验证集，测试集的分布差异较大，及其容易过拟合  
8. __Note：__ 这次比赛的训练集，验证集，测试集的分布差异较大，包括训练集自身不同地震段的分布都不均与（这点可以从不同kfold导致cv差异很大看出）所以防止过拟合尤为重要  
<img src="./earthquake_architectural.png">  


## File Discribe
```
-------- DL
  |      |
  |      |----- config.py: model config
  |      |
  |      |----- dataset_helper.py: 1.eatract fatures 2.define Dataset & DataLoader
  |      |
  |      |----- models.py: 1.define LSTM & CNN models
  |      |
  |      |----- generate_dl_feature.py: train & extract dl features from DL models
  |      |
  |      |----- dl_utils.py
  |      |
  |       ----- adabound.py: adabound optimizer (not use here)
  |
  |----- Statistics
  |      |
  |      |----- dataset.py: eatract Statistics features
  |      |
  |       ----- feature_importance.py: radnom mix features & filter top values
  |
  |----- global_variable.py: global variable
  |
  |----- generate_kfold.py: generate kfold.pkl
  |
  |----- nn.py: train & inference by NN
  |
  |----- lgb.py: train & inference by LGB
  |
  |----- randomforest.py: train & inference by random-forest
  |
  |----- svr.py: train & inference by svr(SVM in regression)
  |
  |----- xgb.py: train & inference by XGB
  |
  |----- ensemble.py: ensemble predictions by LinearRegress
  |
   ----- utils.py: utils functions


```


## Top Rank Solution  
有几个solution比较印象深刻：  
1. 也采用了ks检验，但他不仅用ks检验过滤特诊，还用ks检验过滤训练集，在训练集上抽取与测试集分布相似的子集作为训练集  
2. 因为有数据泄露，所以可以预估测试集的均值，通过把训练集的均值强行拉成测试集的均值，从而达到更好的效果  
3. 加入multitask, 同时预测片段的开始时间，发生时间大于某个阈值的可能性，等等   
4. 大幅度加大cv，第6名用了16fold的cv  