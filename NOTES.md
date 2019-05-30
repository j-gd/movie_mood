# Data gathering & preliminary EDA
## Objectives
Find a large number of movie reviews, 
Including the movie title,
Including multiple reviews per title


## Key hypotheses
Critique reviews such as Roger Ebert have many topics and complex prose, making
it hard to get clear mood, so I will be looking for audience reviews instead.
It's better to have multiple reviews.

##DataSets
### Amazon reviews
http://jmcauley.ucsd.edu/data/amazon/
100K movie and TV shows with title, on or before 2014

#### Good things about the data set
Helpful/not helpful review
Multiple reviews by same person

#### Challenges regarding the data set
Contains seasons and collections
Does not contain the year of the movie
Does not contain the genre

#### EDA
1. Check the distribution of reviews per movie

Potential improvements:
Tag all words indicating collections ('collection', 'series [#]', 'volume [#]',
'double feature', ...)

Top of the list: 'Downton Abbey Season 3', with 12000 reviews
