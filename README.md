# Protein-Structure-Prediction

## Task Introduction
The prediction of protein structure is a very important task in bioinformatics. In this task, we were given the protein sequences with different folds types, and tried to build a machine learning model to classify them.

## Data Introduction
The dataset totally includes **11843** protein sequences with **245** different fold types. And the average similarity among these protein sequences is below 40%. 

The total 11843 sequences were splited into training set and test set with size 9472 and 2371 respectively, and they were stored in 2 separate files `astral_train.fa
` and `astral_test.fa`.


## Modeling Methods
My initial idea was to use neural network like **text_cnn**, But unfortunately, the final performance on test set was poor.Then I tried to use **lstm+cnn**, though the model performance improved, but not much. 

>The main reason why deep learning method didn't work well in this case is the size of training set is too small, thus neural network can not learn well. However,if you face the same problem of limited sample size, one solution you can try is to use pre-trained models. But for this task, it is not allowed. 

Then I decided to do the feature engineering myself and use traditional machine learning method `SVM` to solve this task.the performance is far beyond the neural network. All my code about the implementation using this method is in `main2.ipynb` notebook, if you are interested in the details, you can check it.

## Final Score
My final best score in this task is **0.29936**, and the final rank is 40/1107.
