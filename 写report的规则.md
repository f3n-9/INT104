# Coursework – Lab Report  

## Tasks  

1. Observe the distribution of raw data and analyse what makes the classification of programmes difficult.   
2. Try various transform of data. Visualisation may be helpful. The expected output is to make the samples of the same programme having a similar distribution.   
3. Compare the candidate features and comment on how the resulting clusters are associated with the programme information, given clustering the data into four clusters.   
4. Try to build several classifiers that predict the student’s programme according to mark distribution. Evaluate the performance of classifiers.  

5. Write a report that analyses that which set of features is better than others for the task of predicting student’s programme.  


## Marking Criteria  

### Task 1 & 2 (25 marks)  

The distribution of raw features is observed. (2 marks)   
There are at least 3 transforms being performed for the raw features. (4 marks each,   
in total 12 marks)   
The features are compared via at least one matric. (8 marks)   
A conclusion is drawn from the comparison. (3 marks)  

### Task 3 (25 marks)  

At least three sets of features are clustered. (3 marks each, in total 9 marks)   
For each set of features, there are at least three clustering configurations are tried. (3   
marks each, in total 9 marks)   
There are justifications provided on the experiment design, i.e., the choice of   
clustering methods and configurations. (5 marks)   
The best performed clustering results are compared with the distribution of   
programmes. (2 marks)  

### Task 4 (25 marks)  

At least three sets of features are used for classification of programmes. (2 marks each, in total 6 marks)   
For each set of features, there are at least three sets of hyper parameters are tried. (2 marks each, 6 marks in total)   
An ensembled classifier is built with attempt of 2 sets of different hyper-parameters. (3 mark each, in total 6 marks)   
There are justifications provided on the experiment design, i.e., the choice of classification methods and configurations. (5 marks) 
Comment on why the best choice of features and classifiers. (2 marks)  

### Discussions and Conclusions (10 marks)  

Present a new finding being discovered by the experiment results presented. (4 marks)
Justify how the discovered finding is supported by the experiment results. (3 marks)
Based on the existing results, what further work could be performed to strength the presented new finding. (3 marks) 




### 我的发现

#### Feature Selection 部分
- Programme 1 3成绩好，2 4成绩差 1成绩好且方差小 3成绩好但是方差大
### Average Scores for Each Programme (Q1-Q5 and Total Score)

### Scores for Each Programme (Q1-Q5 and Total Score)

| Programme | Q1     | Q2     | Q3      | Q4     | Q5     | Total_Score |
|-----------|--------|--------|---------|--------|--------|-------------|
| 1         | 7.2910 | 4.9524 | 12.1429 | 7.2725 | 1.6984 | 33.3571     |
| 2         | 6.0909 | 2.5909 | 9.5795  | 4.8295 | 0.7955 | 23.8864     |
| 3         | 6.3846 | 3.8462 | 11.2692 | 7.4231 | 1.6154 | 30.5385     |
| 4         | 6.2025 | 3.2025 | 9.6564  | 4.3282 | 0.7546 | 24.1442     |

### Variances for Each Programme (Q1-Q5 and Total Score)

| Programme | Q1     | Q2     | Q3      | Q4      | Q5     | Total_Score |
|-----------|--------|--------|---------|---------|--------|-------------|
| 1         | 2.1542 | 4.3222 | 11.2082 | 8.6767  | 2.1586 | 53.6337     |
| 2         | 6.8882 | 4.9801 | 18.2005 | 13.8672 | 2.0726 | 78.7685     |
| 3         | 6.7262 | 5.4154 | 19.8046 | 9.9338  | 3.2062 | 100.8185    |
| 4         | 5.7674 | 5.7057 | 18.7454 | 13.9055 | 1.4085 | 90.7399     |
- 取数据集的所有子集，再把子集里数据平均，计算出和programme相关性很大，且column较少的是（Gender, Q2, Q4）相关性=-0.3753(correlation矩阵图片：top_20_subset_programme_correlations.png)

- grade==3 和programme == 3相关性很大（插一张图片：grade3和programme3关系.png），但是手动在分类算法中让grade == 3默认分类programme3，其他重新训练classifier效果并不会提升(在下文会提到)
- pca和tsne对此数据集基本没有效果，可能跟数据集本身的分布有关，pca和tsne只能在数据集有明显分布的情况下才能起到降维的作用（cluster会提到）


#### Unsupervise Learning 部分
- 三组数据集是
  - **Set1**: `Grade`, and `Q_avg` (average of `Q1-Q5`), scaled using MinMaxScaler.
  - **Set2**: `Q1-Q5`, reduced to 2 dimensions using PCA, standardized.
  - **Set3**: All features (`Gender`, `Grade`, `Q1-Q5`), reduced to 2 dimensions using t-SNE, standardized.

其中Set1有最好的inter/intra ratio = 0.1519，其他都在0，25以上


- cluster选择的特征数量少有利于inter/intra ratio变小（极端情况选inter == 0的）尝试了全选，（`Gender`, `Grade`, `Q_avg`），（`Grade`, `Q_avg`）三种，ratio依次变低，但是和真实标签也愈发遥远

|                     | All features | (`Gender`, `Grade`, `Q_avg`) | (`Grade`, `Q_avg`) | (`Grade`, `Gender`) no scale |
|---------------------|--------------|------------------------------|--------------------|------------------------------|
| Inter / Intra Ratio | 0.6897       | 0.2087                       | 0.1519             | 0.0                          |
| F1 Score            | 0.573945     | 0.4520                       | 0.4213             | 0.452025                     |
其他参数是MinMaxScaler，AgglomerativeClustering(n_clusters=4, linkage='complete')

- hierarchical经常能做到最小的inter/intra ratio，但是和真实情况经常相去甚远（f1score低），非常容易过拟合。而kmeans是最不容易的（ratio高，f1score也高）
数据集是(`Gender`, `Grade`, `Q_avg`)，MinMaxScaler

| Method             | Intra/Inter Ratio | F1-Score  |
|--------------------|-------------------|-----------|
| K-means            | 0.265247          | 0.547522  |
| Hierarchical       | 0.208723          | 0.452025  |
| Gaussian Mixture   | 0.264758          | 0.433916  |

#### supervise Learning 部分
目前用的数据Set1是Gender, Q2, Q4（Feature Selection选的相关性较大的一组数据），Set2是All features（Standardized）

- Naive Bayes column太多数据效果会变差（特征全选不如只选部分），其他两个模型column多会变好（全选最好）

| Classifier    | Feature Set（Good） | Accuracy | Feature Set（Bad） | Feature Set |
|---------------|-------------|----------|-------------|-------------|
| Naive Bayes   | Set 1       | 0.5536   | Set 2       | 0.544887    |
| Decision Tree | Set 2       | 0.5601   | Set 1       | 0.5172      |
| KNN           | Set 2       | 0.5729   | Set 1       | 0.5215      |

- Naive Bayes和Decision Tree对scaler不感冒，KNN做了scale成绩会提升（默认用最好的配置）

| Classifier    | Without Scaler | With Standard scaler |
|---------------|----------------|----------------------|
| Naive Bayes   | 0.5536         | 0.5536               |
| Decision Tree | 0.5601         | 0.5601               |
| KNN           | 0.4956         | 0.5729               |

- 观察到只要是成绩好的 classifier算法喜欢把他往Programme1上去分(KNN,所有数据)

| Group                     | Average Total Score |
|---------------------------|---------------------|
| Predicted Programme 1      | 37.090909           |
| Predicted Not Programme 1  | 26.012821           |

- 根据我们的feature obeservasion， 手动在分类算法中让grade == 3默认分类programme3，其他重新训练classifier效果并不会提升，可能因为我不人工分类 机器学习也能学习到这条规律

| Classifier | Normal | 手动预测Programme3 |
|------------|--------|--------------------|
| KNN        | 0.5729 | 0.5606             |

- ensembled classifier 用了all features standardize，然后用了soft和hard，对nb,DecisionTree,knn做了投票，参数是soft和hard。但是由于在all features上nb参数过多，然后decision tree本身性能也不如knn，导致结果不如单独knn
结果 Accuracy

| Classifier       | Accuracy  |
|------------------|-----------|
| Voting (Soft)    | 0.5620    |
| Voting (Hard)    | 0.5428    |
| kNN              | 0.5729    |