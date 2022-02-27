# Credit_Risk_Analysis
## Overview

In order to decide the "best" machine learning model for predicting credit risk, we have trained and evaluated six different models. The data used for these models: `LoanStats_2019Q1.csv`. The idea is to find the ideal model for the job of predicting credit risk.


## Resources
-**`Software:`** [DataSpell](https://www.jetbrains.com/dataspell/) **|** [Jupyter Notebook](https://jupyter.org/install)
-**`Data Source:`** [LoanStats_2019Q1.csv](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv)
-**`Libraries:`** [scikit-learn](https://scikit-learn.org/stable/) **|** [imbalanced-learn](https://imbalanced-learn.org/stable/)



## Results

### Random Oversampling

**`Sampling Method:`** RandomOverSampler()
**`Prediction Method:`** LogisticRegression()

Let's start by looking at our `Random Oversampling` model. The data has been cleaned before being plugged into the model. From there, we used the `LogisticRegression(solver='lbfgs')` linear model to make our predictions.

#### Results:
- Resampled Data Distribution:
  ![Random Over Sampled Data](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/RODataDist.png)


- Balanced Accuracy Score:
  ![Balanced Accuracy Score](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/ROBalancedAccuracyScore.png)


- Confusion Matrix:
  ![Confusion Matrix](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/ROConfusionMatrix.png)

  
- Imbalanced Classification Report:
  ![Imbalanced Classification Report](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/ROImbalancedClassificationReport.png)


With these results in front of us, there are a few things to note before moving on:


1. Our **imbalanced classification report** shows a few different columns but in our "pre" (precision) column, the low_risk precision value is at `1.00 (100%)`.  This means our model has predicted ALL of the low_risk clients correctly BUT with accuracy so high, odds are, our model has also incorrectly predicted some high_risk clients as low_risk.
    2. To confirm that our model is incorrectly predicting high_risk clients as low_risk; We can look at the same `"pre" (precision)` column, but this time at the high_risk value: `0.01 (1%)`. This means that our model only predicted `1%` of high_risk clients correctly! That is not good... Especially if the lenders want to see their money again!
2. With our **balanced accuracy score ** of `64%` our model isn't very accurate.
3. Last thing of note is our **avg / total** value of our precision column. The total precision percentage for our model is 99%. Don't be fooled though; This figure is heavily boosted due to the `low_risk` predictions doing so well.

All in all, I'd say we should keep moving and see if we can find a better model for the job!


### SMOTE  | Synthetic Minority Oversampling Technique

#### `An oversampling technique that generates synthetic samples from the minority class.`

**`Sampling Method:`** SMOTE()
**`Prediction Method:`** LogisticRegression()

Next up is SMOTE! Also known as, Synthetic Minority Oversampling Technique. We will be using the same data that was cleaned prior and then use the same `LogisticRegression(solver='lbfgs')` linear model to make our predictions.

#### Results:
- Resampled Data Distribution:
  ![SMOTE Resampled Data Dist](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/SmoteDataDist.png)


- Balanced Accuracy Score:
  ![SMOTE Balanced Accuracy Score](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/SmoteBalancedAccuracyScore.png)


- Confusion Matrix:
  ![SMOTE Confusion Matrix](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/SmoteConfusionMatrix.png)


- Imbalanced Classification Report:
  ![SMOTE Imbalanced Classification Report](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/SmoteImbalancedClassificationReport.png)


The results of our `SMOTE` model are **very** close to those of our `Random Oversampling` model. This makes sense considering that our `SMOTE` model is employing the power of oversampling, but instead of using existing data points like `Random Oversampling`, it will generate its own synthetic data points; These data points are generated based off of the existing data.

Let's move on and see if any of the other models are more accurate.



### ClusterCentroids Undersampling

#### `Method that under-samples the majority class by replacing a cluster of majority samples by the cluster centroid of a KMeans algorithm. This algorithm keeps N majority samples by fitting the KMeans algorithm with N cluster to the majority class and using the coordinates of the N cluster centroids as the new majority samples.`

**`Sampling Method:`** ClusterCentroids()
**`Prediction Method:`** LogisticRegression()

#### Results:
- Resampled Data Distribution:
  ![ClusterCentroids Data Dist](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/CCUndersamplingDataDist.png)


- Balanced Accuracy Score:
  ![ClusterCentroids Balanced Accuracy Score](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/CCUndersamplingBalancedAccuracyScore.png)


- Confusion Matrix:
  ![ClusterCentroids Confusion Matrix](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/CCUndersamplingConfusionMatrix.png)


- Imbalanced Classification Report:
  ![ClusterCentroids Imbalanced Classification Report](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/CCUndersamplingImbalancedClassificationReport.png)

Just when we thought things couldn't possibly get worse, right?! Even though our `CentroidCluster()` model is definitely not the model for the job, let's talk about a few points of note.

1. I'd say the main item of note here is the **undersampling** aspect of our model. Undersampling causes our datasets to match in size but the larger set is lowered to match the smaller set.
2. As you can see from our `Data Distribution` image above, our model is only taking in 260 `low_risk` data points and 260 `high_risk` data points. This isn't much information for our model to learn from.
3. With our **balanced accuracy score** being `51.3%` along with our `1%` **high_risk precision value**, it is clear that the `CentroidCluster` model is not the way to go. Too many `high_risk` clients would slip through the cracks.



### SMOTEEN | SMOTE + Edited Nearest Neighbours
#### `Over-sampling using SMOTE and cleaning using ENN. Combine over- and under-sampling using SMOTE and Edited Nearest Neighbours.`

**`Sampling Method:`** SMOTEENN()
**`Prediction Method:`** LogisticRegression()

Now this model is interesting! As you can see below, for the first time our data is distributed unevenly. This time we have MORE `high_risk` data points and `low_risk` for our model to learn from. This seems like a great step in the right direction if you ask me! Let's see if **SMOTEEN** is better than the previous models.

#### Results:
- Resampled Data Distribution:
  ![SMOTEEN Data Dist](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/ENNDataDist.png)


- Balanced Accuracy Score:
  ![SMOTEEN](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/ENNBalancedAccuracyScore.png)


- Confusion Matrix:
  ![SMOTEEN Confusion Matrix](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/ENNConfusionMatrix.png)


- Imbalanced Classification Report:
  ![SMOTEEN Imbalanced Classification Report](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/ENNImbalancedClassificationReport.png)

Definitely better! The **SMOTEEN** model is a huge improvement from our `ClusterCentroid()` model according to our `balanced accuracy scores`. Compared to our `SMOTE` and `Random Oversampling` models, there is a slight improvement. Let's talk a little more about the results above:

1. The main item of note for our **SMOTEEN** model is the increased `rec (sensitivity)` value. This value of `.71` indicates our sensitivity level. If we compare the **SMOTEEN** `rec` value to our **SMOTE** models `rec` value, you can see an increase of almost `.1`! This is pretty huge and very important. Since we don't want lenders giving away free money, it would be better to accidentally deny an eligible (low_risk) person a loan than accept a high_risk client. This is where sensitivity comes into play.
2. I'd also like to mention again that the `balanced accuracy score` increased by 1%! Nothing major. So far out of the four models we have looked at, **SMOTEEN** is looking like the "best" option.


### BalancedRandomForestClassifier Model
#### `A balanced random forest randomly under-samples each bootstrap sample to balance it.`


**`Sampling Method:`** BalancedRandomForestClassifier()
**`Prediction Method:`** BalancedRandomForestClassifier()

- Feature Importance Score:
  ![Forest Feature Importance Scores](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/ForestFeatureImportance.png)


- Feature Importance Chart:
  ![Forest Feature Importance Chart](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/Importance.png)


- Balanced Accuracy Score:
  ![Forest Balanced Accuracy Score](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/ForestBalancedAccuracyScore.png)


- Confusion Matrix:
  ![Forest Confusion Matrix](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/ForestConfusionMatrix.png)


- Imbalanced Classification Report:
  ![Forest Imbalanced Classification Report](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/ForestImbalancedClassificationReport.png)
  

Let's dive right into it!
1. First let's talk about our **balanced accuracy score:** `73.3%`. According to this score, compared to the other models, our **BalancedRandomForestClassifier** model is an improvement. Let's look at our **imbalanced classification report** to find out if this is true.
2. First we will look at our `pre (precision)` values. `low_risk` has stayed consistent across all models with a precision score of `1.00 (100%)`. We are interested in the `high_risk` value though: `0.02 (2%)`. Not exactly what we are looking for. As mentioned earlier, lenders cannot afford low precision levels like this; Otherwise `high_risk` clients would be lent money they probably can't pay back.
3. Lastly, using **BalancedRandomForestClassifier** allows us to view the level of importance each feature offers to the model. Above you can see how heavily each feature is weighed.

Moving on to our final model: `EasyEnsembleClassifier`


### EasyEnsembleClassifier Model
#### `An ensemble of AdaBoost learners trained on different balanced bootstrap samples. The balancing is achieved by random under-sampling.`


**`Sampling Method:`** BalancedRandomForestClassifier()
**`Prediction Method:`** BalancedRandomForestClassifier()

Last but definitely not least.... The **EasyEnsembleClassifier** model. Right off the bat we can see massive improvements! Take a look over the results and let's discuss the data a little further.


- Balanced Accuracy Score:
  ![EEC Balanced Accuracy Score](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/EnsembleBalancedAccuracyScore.png)


- Confusion Matrix:
  ![EEC Confusion Matrix](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/EnsembleConfusionMatrix.png)


- Imbalanced Classification Report:
  ![EEC Imbalanced Classification Report](https://github.com/StickySitch/Credit_Risk_Analysis/blob/main/Images/EnsembleImbalancedClassificationReport.png)


1. With a **balanced accuracy score** of `.93 (93%)`, it is clear that out of the six, the **EasyEnsembleClassifier** model is the best of the six. Let's dive into the `imbalanced classification report` and see if this model is truly right for the job of predicting credit risk.
2. When looking at our **EasyEnsembleClassifier** models `imbalanced classification report`, we can see that our `high_risk` row has a **pre (precision)** of `0.9` and a **rec (sensitivity)** of `.92`. A higher sensitivity is exactly what we are looking for! Unfortunely, this increased sensitivity is not enough to make our model precise.


## Summary

We have gone over a lot here but the question is; **Which model should be used to predict credit risk?**
The answer is **none** of them. Although our **EasyEnsembleClassifier** showed promise, due to the low precision score, too many `high_risk` clients would be approved for loans. With **EasyEnsembleClassifier** being the "best" option and still falling short; It is clear that I cannot recommmend any of the other models, as they all fall short of the **EasyEnsembleClassifier** model.


