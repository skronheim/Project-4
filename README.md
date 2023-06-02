# Project-4

## Overview

This project is based on [Spotify data](https://developer.spotify.com/documentation/web-api). 

Songs will be clustered based on their characteristics via K-means, and the clustering will be used to generate music recommendations based on an input song. The goal is to provide 5-10 song recommendations of similar songs for any input. 

If time permits, we will further train a machine learning model to predict the popularity of an input song based on its characteristics. 

### Notes from 01Jun23
collaborative filtering - correlations between you and other people with things you like, recommend songs based on similar tastes
- item item metrics - item similarity
- user similarity metrics - user similarity
--amazon, netflix, etc
-- movie/ music/ etc recommendations, based on ratings by other users?
- rating, or times played, or likes

- can do recommendations based on type of song

- Spotify dataset
-- K-Means to cluster songs based on characteristics and return recommendations based on an input song
-- Neural network to figure out whether a song is likely to be popular; should take in the original input song as well as the recommendations