# Online-Challenge-Build-A-Recommendation-Engine-Powered-by-IBM-Cloud-
Analytics Vidhya hackathon, rank 36 on leader board from 3978 registered participants, 186 submitted to leader board.
## Libraries
mlxtend, numpy, pandas
## Files
recomendation_f.py - code; train_5UKooLv.csv - train data; test_J1hm2KQ.csv - test data

## Author
Inga Kuznetsova

## Description
The recommendation of popular products worked out better than other simple methods. I used my own function just to count what items are bought often by different customers. The customers are divided by groups by their activity. More active customers, who bought more items, get more recommendations, but at the end items that were bought before were removed from predicted. I added seasons. The data were divided by 4 seasons. I recommended popular items for seasons when given customer did shopping. This worked better on more active customers.  I also added recommendations using apriori library in some groups, where it worked better: changed my most popular to items with higher apriori score, added recommendations to buy item based on was bought before, based on pairs of items score.
