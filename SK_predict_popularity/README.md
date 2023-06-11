# Model Training and Optimization

## Table of Contents

1. Neural Network - Categorical Bins
2. Neural Network - Binary Bins
3. Summary

---

### Neural Network - Categorical Bins

* I first tried converting popularity into categorical bins and predict the bin based on song characteristics. The accuracy and loss are in the summary_stats folder. I tested several different preprocessing methods and then performed hyperparameter tuning to determine the best model structure once I settled on the preprocessing.
* I tried the iterations below:
    1. Standard scaler with 4 popularity bins:
        * 0 = "None" = 0
        * 0 < popularity <= 30 = "Low" = 1
        * 30 < popularity <= 60 = "Medium" = 2
        * popularity > 60 = "High" = 3
    2. Standard scaler with 3 popularity bins:
        * popularity <= 30 = "Low" = 0
        * 30 < popularity <= 60= "Medium" = 1
        * popularity > 60 = "High" = 2
        * The 3-bin model improves the accuracy and decreases the loss, so 3-bin was carried forward
    3. Min max scaler with 3 popularity bins
        * Both accuracy and loss improved; accuracy increased by ~1% and loss decreased by ~2%
        * MinMax scaling was carried forward
    4. Random oversampling with 3 bins and MinMax scaling 
        * The data is very skewed toward unpopular; the accuracy of the initial model could result from the model calling everything unpopular
        * Oversampling was performed to reduce skew in the dataset
        * This decreased accuracy and increased loss
        * Still keep this, because the data is very skewed and initial accuracy could almost be the result of calling everything unpopular
    5. Normalization only on subset of columns with oversampling, 3 bins, and MinMax scaling
        * This decreased the accuracy very slightly and increased the loss slightly; moving forward the whole dataset was normalized
    6. Hyperparamerter tuning on iteration 4
        * Increased the accuracy to 66.7% with loss of 97%

---

### Neural Network - Binary Bins

* Since changing from 4 bins to 3 bins increased the accuracy significantly, I also tried converting popularity into a binary:
    * < 60 popularity = Unpopular = 0
    * > 60 popularity = Popular = 1
* With these binary bins, I tested similar iterations to those described above.
    1. StandardScaler normalization
    2. MinMaxScaler
        * Accuracy increased slightly while loss decreased slightly
        * I switched to MinMax scaling for normalization
    3. Random oversampling with MinMax scaling
        * The data is very skewed toward unpopular; the accuracy of the initial model could result from the model calling everything unpopular
        * Added random oversampling to account for the skew 
        * Accuracy decreased
        * Still keep this, because the data is very skewed and initial accuracy could almost be the result of calling everything unpopular
    4. Normalization only on subset of columns with otherwise the same preprocessing as iteration 4
        * Accuracy increased
    5. Hyperparameter tuning with the iteration 4 pre-processing steps
        * Increased the accuracy

---

### Summary

* Binary classification of popular vs not has much higher accuracy than categorical classification with a sequential neural network model