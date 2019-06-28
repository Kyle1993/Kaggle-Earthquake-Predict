# Kaggle Earthquake Predict

__Competitions:__ [LANL Earthquake Prediction](https://www.kaggle.com/c/LANL-Earthquake-Prediction)  
__Rank:__ 186/4540 (Top 4%, silver)  
__Note:__ This is a code backup, it's not runable due to the difference file path  



## Architectural
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

## Note
1. Use ks_2samp & correlate check to filter features, avoid overfitting (refer utils.py)
2. Use signal process skill to extract features (refer Statistics/dataset.py)