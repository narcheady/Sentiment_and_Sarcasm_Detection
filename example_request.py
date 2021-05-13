from review_classifier_service import ReviewClassifierService
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
tweetsDf = pd.read_csv('twitter_train.csv')
#tweetsDf = pd.read_csv('tweetsViaAPI_merged_with2021.csv')
#tweetsViaAPI
#print(tweetsDf.head(10))

service = ReviewClassifierService()

positive_sample = "this is the worst"
#negative_sample = "facebook reportedly working on healthcare features and apps"
negative_sample = "this has been a very positive news in times on covid"
#print(service.sentiment_classify(positive_sample))
#print(service.sentiment_classify(negative_sample))


rowlist = []
for row in tweetsDf.head(20).itertuples():
    rowlist.append([row.text,service.sentiment_classify(row.text), service.sarcasm_classify(row.text)])

resultsDf = pd.DataFrame(rowlist,columns = ['text', 'sentiment', 'sarcasm'])
print(resultsDf)