# Movie Mood: finding emotions in movies via reviews

## Objective

To extract emotions contained in movies, from movie reviews, filtering out plot descriptions, reviewers emotions about the movie, and comments about the support (Blu-ray…) or shipping. This is the foundation for a search engine that would allow users to select a movie based on the emotions found in the movie by reviewers.

## How the objective was achieved

Given a review such as: ‘<span style="color:red">Great movie!</span><span style="color:green"> Takes place in an isolated outpost in the galaxy.</span> <span style="color:blue">The hero hates aliens.</span> <span style="color:gray">The Blu-ray contains awesome bonus material.</span>’, the goal was to isolate: ‘<span style="color:blue">the hero hates aliens</span>’.

#### 1. Removed support-related sentences
Removed support-related sentences (e.g. ‘<span style="color:gray">The Blu-ray contains awesome bonus material</span>’) via keyword search.

#### 2. Removed descriptive sentences
Removed descriptive sentences (‘<span style="color:green">takes place in an isolated outpost in the galaxy</span>’) by vectorizing text into a space of 7 emotions and removing sentences below a threshold.

#### 2. Removed reviewers' feelings

Removed reviewers’ feelings by modeling the differences between feeling descriptions in plot (‘<span style="color:blue">the hero hates aliens</span>’) and the reviewer’s feelings (‘<span style="color:red">great movie!</span>’).

## Other findings
I used the star ratings associated with the reviews as a way to validate the models I developed. I found that feelings expressed in reviews (‘<span style="color:red">great movie!</span>’ & ‘<span style="color:blue">the hero hates aliens</span>’) correlate positively with ratings:

Reviews with strong scared feelings have lower ratings than others:
![Fear level of movies with top 5% fear review content, vs. others](./images/high_fear_content.png)

Reviews with strong happy feelings have higher ratings than others:
![Fear level of movies with top 5% happy review content, vs. others](./images/high_happy_content.png)

It makes sense that reviewers' feelings correspond to ratings. However, emotions in movies should be less related to reviewers' ratings, as there are good scary movies out there!

So I built a classifier to differentiate reviewer feelings from plot sentences, and built a sentiment predictor model to test the classifier. I computed the accuracy of the sentiment analysis model after the classifier removed either reviewer feelings or plot sentences. I found that removing sentences with reviewer feelings (<span style="color:blue">blue bars</span> below) reduced the accuracy of the sentiment predictor much more than by removing plot-related sentences (<span style="color:red">red bars</span> below): 

![Accuracy of sentiment predictor as sentences are removed](./images/accuracy_sents_removed.png)

It makes sense: reviewers' feelings are more connected to ratings than feelings in reviews are.

## Technical details

### Data set

I obtained 4.6 million Amazon movie & TV reviews from J. McAuley of UCSD, which he collected for the research paper: *Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering
by R. He, J. McAuley, WWW, 2016 [pdf](http://cseweb.ucsd.edu/~jmcauley/pdfs/www16a.pdf)*

### Bag of words model with 7 emotions

I used a dataset of 23,000 keywords labeled with the following 7 emotions: 'disgust', 'surprise', 'anger', 'sad', 'happy', 'fear' and 'neutral'.

### Separating reviewer’s feelings from feelings in plot descriptions

#### Reviewer feelings vs. plot classifier
Used 5,000 sentences from a plot description site, and 5,000 sentences from rotten tomatoes for the labeled data. Cleaned up the labels manually by inspecting data misclassified by the models.

#### Sentiment classifier
Created it to validate the reviewer/plot classifier.

Isolated a set of 30,000 movie reviews with 5 sentences and balanced +/- sentiment, from the Amazon data set.

#### For both classifiers
Performed grid searches with logistic regressors, random forests and gradient boosting classifiers (GBC). Found best results with GBC.

## Slide presentation

A slide presentation of this readme file is available [here](https://drive.google.com/file/d/1CpffON2RjL-idEwLhmM73ZFLLSwfjmu8/view?usp=sharing).

## How to run unit tests

In the project root directory, run: pytest test/unittests.py
