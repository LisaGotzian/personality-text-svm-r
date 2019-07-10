# Predicting MBTI personalities using text and SVMs in R

This analysis intends to (1) predict MBTI types based on social network posts as done with the Big 5 to then (2) determine if the MBTI is a valid measure for text data. The results show that text can indeed be a good predictor for people’s MBTI personality types. The analysis also showed that similar personality types tend to share communities and hang out with each other. This network effect can be used for future research and implies that network analysis of social media profiles might also predict personality types well.  
`MBTI_Personality.pdf` explains the procedure and the academic background to this analysis.

## The data
The data in `mbti_1.csv` consists of 8675 rows of text, each with 50 posts or comments including the MBTI type of said person. It has been scraped from the ”[Personality Cafe forum](http://personalitycafe.com/forum/)” and made [publicly available](https://www.kaggle.com/datasnaek/mbti-type).

## The SVM
Support Vector Machines (SVMS) are known to produce stable classification results based on text because they hardly overfit. Additionally, text is usually linearly separable, allowing SVMs to determine a margin between categories ([Joachims, 1998](http://link.springer.com/10.1007/BFb0026683)). For more details on the method, read `MBTI_Personality.pdf`.

## Results
| Model |Accuracy|
|-------|------|
|plain features | 0.21 |
|plain text |0.56 |
|text + features | 0.62 |
|**text + features +** ||
|unbiased w/o type | 0.46 |
|robust 7 classes | 0.54 |
robust + unbiased | 0.54 |


## Built with
* RTextTools
* ngram
* tm
* quanteda
* caret
