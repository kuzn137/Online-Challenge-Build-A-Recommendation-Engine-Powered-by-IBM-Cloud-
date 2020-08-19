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

## License
Copyright 2020 Inga Kuznetsova

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
