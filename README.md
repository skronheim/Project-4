# Project-4

# Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
    1. [Description](#description)
    2. [Limitations](#limitations)
3. [Data Preprocessing](#data-preprocessing)
4. [Models](#models)
    1. [Unsupervised (K-means)](#unsupervised-k-means)
    2. [Supervised (Neural Networks)](#supervised-neural-networks)
        1. [Popularity Prediction](#popularity-prediction)
        2. [Genre Prediction](#genre-prediction)
5. [Summary](#summary)

# Overview

the aim of our project is to group songs by their characteristics in order to create a song recommendation program. We also tried to predict the popularity and genre of songs based on their characteristics. We used K-means to cluster the songs by their characteristics, and neural networks to predict their popularity and genre. 

# Dataset

## Description

This project is based on a [Spotify dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) that was downloaded from Kaggle. 

The included metrics and their values are as follows:
* Popularity - 0-100; This is a representation of the song's popularity
* Duration - Wide range of numbers; this describes the duration of the song in miliseconds (ms)
* Explicit - T/F; This describes whether the lyrics are explicit
* Danceability - 0-1; This describes how suitable a song is for dancing based on factors including tempo, beat strength, and overall regularity
* Energy - 0-1; This is a perceptual measure of energy and activity in the song
* Key - 0-11; This is the key 
* Loudness - ~-49-4; This is the overall loudness of the song, in decibels (dB)
* Mode - 0 or 1; This describes whether the song is in major or minor key - 0=minor key, 1=major key
* Speechiness - 0-1; Measure of how much spoken word is present in a song
* Acousticness - 0-1; A confidence measure of whether a song is acoustic
* Instrumentalness - 0-1; This is a prediction of whether a song has vocals
* Liveliness - 0-1; This is a prediction of whether the track was recorded live (the presence of an audience in the recording)
* Valence - 0-1; Describes how positive the music sounds
* Tempo - 0-243; Beats per minute (BPM)
* Time signature - 3-7; This represents time signatures from 3/4 - 7/4, with the value divided by 4 to obtain the time signature
* Track_genre - Genre; 114 genres are included

## Limitations

This dataset was downloaded from Kaggle, and consists of data collected using the [Spotify API](https://developer.spotify.com/documentation/web-api).

* The biggest limitation of this dataset is genre
    * Spotify does not track genre specifically for songs; it tracks genre for artists and albums
        * The genre data here was collected through API calls looking for each specific genre and recording what songs (with their characteristics) were returned
        * This means that some of the songs are duplicates that had to be filtered out
        * This also means that some of the songs should have multiple genres but do not in this dataset
* Another limitation is the size of the dataset. The data consists of ~1000 songs for each of the 114 genres, but some of thees are duplicates. There are over 100,000 songs in this dataset, but more could make the models more robust

# Data Preprocessing

**@Group - Please add any preprocessing you did for your models that were not done in my data cleaning notebook, and read over the data processing section in general! I have pointed out a few areas that need your input in particular below**

Each model used slightly different preprocessing, as detailed below. 
* Duplicate songs were removed
* Songs with a duration of <= 0 were removed
* The number of artists for each song was determined, and the number of artists was binned
    * This data was used only in the popularity prediction model
    * The bins made this data categorical, so it was later encoded
* The primary artist for each song was determined

    **@Brenda - Did you use primary artist? If you didn't, just remove the bullet point above since neither I nor Carly used artist**

* Track ID was removed
* Duration was converted to minutes
* Categorical data was encoded
    * Time signature and key were one-hot encoded
* All T/F data was converted to integer type (to ensure numerical data)
* Model-specific preprocessing:
    * Data was normalized
        * For the popularity prediction, StandardScaler and MinMaxScaler were tested, as well as MinMaxScaler only on the Loudness, Tempo, and Duration data (the only data that was not either binary or between 0 and 1)

        **@Group - Add your normalization methods here; the order should be Brenda, then Sarah, then Carly to keep consistent with the proposed ppt order (K-Means -> popularity -> genre)**

    * The data pertaining to the target variable for the supervised learning models was manipulated
        * For the popularity prediction, popularity was binned into 2, 3, or 4 segments and encoded using the to_categorical method for the 3 and 4 bin models

        **@Carly - Add any manipulations you made on the genre data here! This should include encoding each genre with a number**

    * Subsets of the data were removed
        * For the popularity prediction the Artist, Album, and Track Name data was removed

        **@Group - Add any data you removed or kept here (whichever list is shorter); the order should be Brenda, then Sarah, then Carly to keep consistent with the proposed ppt order (K-Means -> popularity -> genre)**

    * Further data was encoded
        * For the popularity model, genre was binary encoded

        **@Group - Add any extra data encoding you performed here, if any (if there is none, that's fine); the order should be Brenda, then Sarah, then Carly to keep consistent with the proposed ppt order (K-Means -> popularity -> genre)**

# Models

## Unsupervised (K-Means)

## Supervised (Neural Networks)

### Popularity Prediction

* We first tried converting popularity into categorical bins and predict the bin based on song characteristics. The accuracy and loss are in the summary_stats folder. We tested several different preprocessing methods and then performed hyperparameter tuning to determine the best model structure once we settled on the preprocessing.
* We tried the iterations below:
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
    3. Min-max scaler with 3 popularity bins
        * Both accuracy and loss improved; accuracy increased by ~1% and loss decreased by ~2%
        * MinMax scaling was carried forward
    4. Random oversampling with 3 bins and MinMax scaling 
        * The data is very skewed toward unpopular
        * Oversampling was performed to reduce skew in the dataset
        * This decreased accuracy and increased loss
        * Still maintained this preprocessing step, because the data is very skewed and initial accuracy could almost be the result of calling everything unpopular
    5. Normalization only on subset of columns with oversampling, 3 bins, and MinMax scaling
        * Loudness, tempo, and duration were normalized
        * This decreased the accuracy very slightly and increased the loss slightly; moving forward the whole dataset was normalized
    6. Hyperparamerter tuning was performed to determine the optimal number of hidden layers and nodes in each layer with popularity binned into 3 bins, min-max scaling, random oversampling, and normalization of the full dataset
        * A summary of accuracy and loss for each model can be found in "SK_predict_popularity/summary_stats/categorical_popularity_stats.csv"

* Since changing from 4 bins to 3 bins increased the accuracy significantly, we also tried converting popularity into a binary:
    * < 60 popularity = Unpopular = 0
    * > 60 popularity = Popular = 1
* With these binary bins, we tested similar iterations to those described above.
    1. StandardScaler normalization
    2. MinMaxScaler
        * Accuracy increased slightly while loss decreased slightly
        * we switched to MinMax scaling for normalization
    3. Random oversampling with MinMax scaling
        * The data is very skewed toward unpopular
        * Added random oversampling to account for the skew 
        * Accuracy decreased
        * Still maintained random oversampling, because the data is very skewed and initial accuracy could almost be the result of calling everything unpopular
    4. Normalization only on subset of columns with otherwise the same preprocessing as iteration 4
        * Loudness, tempo, and duration were normalized
        * This increased the model accuracy
    5. Hyperparameter tuning was performed to determine the optimal number of hidden layers and nodes in each layer with min-max scaling, random oversampling, and normalization of only loudness, tempo, and duration
        * A summary of accuracy and loss for each model can be found in "SK_predict_popularity/summary_stats/binary_popularity_stats.csv"

### Genre Prediction

# Summary

In terms of popularity predictions, a neural network model is able to predict the binary popular vs unpopular categories relatively well, with a final accuracy of ~83%. However, the model is significantly less accurate the more segments the popularity is split into, with accuracy of only 67% with four bins and 71% with three bins (before the addition of random oversampling, such that these accuracy scores may be too high). Random oversampling in all cases decreased the model accuracy. However only 12% of the songs were labeled with popularity > 60, so the higher accuracy may not be fully representative without the oversampling. Overall while we were able to build a relatively accurate model to predict the popularity of a song based on its characteristics, this dataset does not support this prediction very well due to how skewed the dataset is toward 'unpopular' songs. This may be a problem with this specific dataset, that the inclusion of more songs may improve, or it may be a problem with Spotify's measure of popularity (perhaps popularity is very stringent, such that very few songs are actually marked as popular). 

**@Group - we'll want to tie everything together here, and I can't fully do so without your data/ analysis. Brenda - you'll want to say something to summarize your KMeans; Carly, you'll want to say something along the lines of 'this data does not support genre prediction very well and here's a possibility as to why'. My material is here, but will need to be fit into whatever you two have to say**