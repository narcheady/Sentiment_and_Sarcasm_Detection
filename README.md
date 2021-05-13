# Sentiment and Sarcasm Detection using Twitter Dataset

This project explores different classification algorithms to classify Twitter data based on Sentiment and Sarcasm. Following steps are performed step-by-step :</br>
<ul>
<li>Twitter Data is being imported using the Twitter api python module. The dataset consists of tweet_ids, tweet_texts, usernames and date.</li>
<li>Perform Exploratory Data Analysis on the dataset, cleaning out unnecessary values, visually plotting our results.</li>
<li>Perform Sentiment Analysis using Textblob's inbuilt sentiment function and also by fitting a model on a labeled dataset and then predicting the sentiment using the model.</li>
<li>Detect Sarcasm by training our model based on a labeled dataset, then predicting the sarcasm in our tweets.</li>
<li>Compare F1 scores of classification models.</li>
