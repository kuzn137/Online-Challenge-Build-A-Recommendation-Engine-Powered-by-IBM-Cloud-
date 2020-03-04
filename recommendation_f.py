# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 10:17:50 2019

@author: kuzn137

Online Challenge: Build A Recommendation Engine (Powered by IBM Cloud)
"""
#from recommendation_ff import recommendation_ff
import pandas as pd
import numpy as np
import warnings; warnings.simplefilter('ignore')
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

def remove_extra(vect, a):
###REMOVE RECOMMENDED POPULAR ITEMS WHICH REPEAT USER ITEMS #####
    if a in vect:# and a not in b:
       vect.remove(a)
    return vect
def buy_again(vect, a, b):
####recommend items that usually bought more than ones to users who bought this item before########################
    if a not in vect and a in b:
       vect.append(a)
    return vect
class recom_seasons():  
      def __init__(self, df):
          self.df = df             
      def find_popular(self, n):
###############find n most popular items#############
          popularity= self.df['StockCode'].value_counts().to_frame().reset_index().rename(columns={'index':'StockCode', 'StockCode':'count'})
          return popularity['StockCode'].iloc[0:n].tolist()
      def find_popular_by_cust(self, n):
###############find n most popular items by how many customers bought it###################
          df1=self.df[['StockCode', 'CustomerID']].drop_duplicates()
          popularity= df1['StockCode'].value_counts().to_frame().reset_index().rename(columns={'index':'StockCode', 'StockCode':'count'})
          return popularity['StockCode'].iloc[0:n].tolist()
      def seasons(self):
    ######split data frame by 4 seasons##########
          self.df['InvoiceDate']=pd.to_datetime(self.df['InvoiceDate'])
          self.df['InvoiceDate']=self.df['InvoiceDate'].astype(str)
          self.df['year']=self.df['InvoiceDate'].apply(lambda x: x[0:4])
          self.df['year']=self.df['year'].replace('2010', '0')
          self.df['year']=self.df['year'].replace('2011', '12')
          self. df['year']=self.df['year'].astype(int)
          self.df['month']=self.df['InvoiceDate'].apply(lambda x: x[5:8].replace('-0', ''))
          self.df['month']=self.df['month'].apply(lambda x: x.replace('-', ''))
          self.df['month_tot']=self.df['year']+self.df['month'].astype(int)
          self.df['season']=0
          self.df.loc[self.df['month'].isin(['03', '04', '05']), ['season']]= 1
          self.df.loc[self.df['month'].isin(['06', '07', '08']), ['season']]= 2
          self.df.loc[self.df['month'].isin(['09', '10', '11']), ['season']]= 3
          return self.df
     
      def split_seasons(self, n, pop_s): 
    #######popular items to recommend by seasons#######################
    #############pop_s list of popular items for each season, 
    ################n is related to numbers of popular items to leave for given activity group###############
              y=[-60, -10, 20, 50]
              for i in range(4):
                  self.df.loc[self.df['season']==i, ['Items']]=self.df.loc[self.df['season']==i, ['Items']].apply(lambda x: x+pop_s[i][:n+y[i]])
              return self.df
      def group_recom_seasons(self, l,  m, pop):
    ####recommendations by seasons###########
    ########m is maximum number of items customer bought############
    #######l is number relatated to recommendations per season#####
    #######pop is list of maximum e=recommendations per season
               dftemp = self.df[(self.df['item_count']<m)]
               dftemp['Items']=dftemp[[]].values.tolist()
               dftemp=recom_seasons(dftemp).split_seasons(l, pop)#pop_winter_act, pop_spring_act, pop_summer_act, pop_fall_act)
               self.df.loc[self.df['item_count']<m]=dftemp               
               return self.df
      def group_recom(self, l,  m):
    ######recommendation based on item popularity###########
               dftemp = self.df[(self.df['item_count']<m)]
               dftemp['Items']=dftemp[[]].values.tolist()
               dftemp['Items']=dftemp['Items'].apply(lambda x: list(set(list(x)+list(l))))#pop_cust[0:250]
               self.df.loc[self.df['item_count']<m]=dftemp
               return self.df
def pop_by_season(df, n):
    ##################find n popular items by seasons##################
            pop_seas=4*[n*[]]
            for i in range(4):
                 pop_seas[i]=recom_seasons(df[df['season']==i]).find_popular_by_cust(n)
            return pop_seas
def scores(df, r1):
    ###################scores from apriori library#################
    ###################r1 is minimum score###########################
    dfst =  df[['CustomerID', 'StockCode']].groupby('CustomerID', as_index=False).agg(list)
    data= dfst['StockCode'].tolist()
    te = TransactionEncoder()
    te_ary = te.fit(data).transform(data)
    dfn = pd.DataFrame(te_ary, columns=te.columns_)
    apr=apriori(dfn, min_support=r1,  use_colnames=True)    
    return apr
def association(apr, r, r1):
    #########select most popular items by score > r including also items which score >r1 but can be inhanced to r if they go together with other item###########
    ###########return data frame a12 with items pair items which has score togeter > r, single items, aprup, with score > r,
    ################items, apr_item1, apr_item2 which together have score >1####
    apr['length'] = apr['itemsets'].apply(lambda x: len(x))
    ####data frame with one item#####
    apr1=apr[apr['length']==1]
    apr1['itemsets'] = apr1['itemsets'].apply(lambda x: list(x))#apr1['item1'] = apr1['itemsets'].apply(lambda x: x[0])
    apr1_up = apr1[apr1['support'] >= r]
    apr1_up['item1']=apr1_up['itemsets'].apply(lambda x: x[0])
    ####single items with score >r#############
    aprup=apr1_up['item1'].values.tolist()
    #####items with lower r>score>r1 which can increase thier score in pair with item with higher score#####
    apr1_low = apr1[(apr1['support'] < r) & (apr1['support'] >= r1)]
    apr1_low['item2'] = apr1_low['itemsets'].apply(lambda x: x[0])
    apr2=apr[apr['length']==2]
    apr2['itemsets'] = apr2['itemsets'].apply(lambda x: list(x))
    apr2['item2'] = apr2['itemsets'].apply(lambda x: x[1])
    apr2['item1'] = apr2['itemsets'].apply(lambda x: x[0])
    apr12 = pd.merge(apr1_up, apr2, on='item1', how='left')
    apr12['conf2'] = apr12['support_y']/apr12['support_x']
    apr12=apr12[apr12['conf2']>=r]
    apr12 = apr12[['item1', 'item2', 'conf2']]
    apr12 = pd.merge(apr12, apr1_low['item2'], on='item2', how='inner')
    apr12 = apr12[['item1', 'item2']].dropna()
    apr12 = apr12.groupby('item1', as_index=False).agg(list)
    apr_item2 = apr12['item2'].tolist()
    apr_item1 = apr12['item1'].tolist()
    apr_item2 =  [item for sublist in apr_item2 for item in sublist]
    return apr12, aprup, apr_item2, apr_item1


def group_recom_apr(df1, apr, m, l, r, r1, typ):
    assoc, apr1_up, apr_item2, apr_item1 = association(apr, r, r1)
    dftemp = df1[(df1['item_count']<m)]
    dftemp['Items']=dftemp[[]].values.tolist()
    l1 = assoc['item1'].tolist()
    l2=assoc['item2'].tolist()
    for i in range(len(l1)):
        dftemp1=dftemp.loc[dftemp['StockCode'].isin([l1[i]])]
        it1 = l2[i]
        dftemp['Items'] = dftemp['Items'].apply(lambda x: x + it1)
        dftemp.loc[dftemp['StockCode'].isin([l1[i]])] = dftemp1
    if typ == 'seasons':
       dftemp=recom_seasons(dftemp).split_seasons(l, pop_seasons)
    else:
       dftemp['Items']=dftemp['Items'].apply(lambda x: list(set(list(x)+list(apr1_up))))
    df1.loc[df1['item_count']<m]=dftemp
    return df1
#####read data##########################
dftr=pd.read_csv('train_5UKooLv.csv', index_col=False)
dft = pd.read_csv('test_J1hm2KQ.csv', index_col=False)

#######add 4 seasons################
dft = recom_seasons(dft).seasons()
dftr = recom_seasons(dftr).seasons()
####Up to 500 most popular items by measure#####
pop_cust = recom_seasons(dftr).find_popular_by_cust(500)
items=dftr.StockCode.unique()
dftr=dftr[['CustomerID', 'UnitPrice', 'Quantity', 'InvoiceDate', 'Country','StockCode', 'season', 'month_tot', 'month']]
dftr_count=dftr[['Country', 'StockCode']].groupby('Country').count()
pop = recom_seasons(dftr).find_popular(350)
dfpop=dftr.loc[dftr['StockCode'].isin(pop), ['StockCode', 'CustomerID']].drop_duplicates()
dfitems=dfpop[['StockCode', 'CustomerID']].groupby('StockCode', as_index=False).count().rename(columns={'CustomerID':'count'})
popularity= dftr['StockCode'].value_counts().to_frame().reset_index().rename(columns={'index':'StockCode', 'StockCode':'count'})
dftr=pd.merge(dftr, dfitems['StockCode'], on='StockCode', how= 'right')

dftr=dftr[['CustomerID',  'UnitPrice', 'Quantity', 'InvoiceDate', 'Country',  'StockCode', 'season', 'month_tot', 'month']]
dft=dft[['CustomerID',  'UnitPrice', 'Quantity', 'InvoiceDate', 'Country', 'StockCode', 'season', 'month_tot', 'month']]
df = pd.concat([dftr, dft])
time = df[['CustomerID', 'month_tot']].groupby('CustomerID').max().rename(columns={'month_tot':'last_time'})
df=pd.merge(df, time, on='CustomerID', how='left')
n_pop_max=400
###up 400 popular in different seasons 
pop_seasons=pop_by_season(df, n_pop_max)
df_act=df[['CustomerID', 'StockCode']].groupby('CustomerID', as_index=False).count().rename(columns={'StockCode':'item_count'})
df =pd.merge(df, df_act, on = 'CustomerID', how= 'left')
df=df[['CustomerID','StockCode', 'UnitPrice', 'Quantity', 'Country', 'item_count', 'month', 'last_time', 'season']]
#####active_customers_who bought more than 400 items############
df_act_cust = df[df['item_count']>=400]
##############items popular among active customers###########
act_cust_pop = recom_seasons(df_act_cust).find_popular_by_cust(340)
##############items popular among active customers by seasons##########################
pop_act=pop_by_season(df_act_cust, n_pop_max)
df1=df[['CustomerID', 'Country', 'season', 'item_count']].drop_duplicates()
####scores to add apriori############
mini = [0.2, 0.189, 0.191, 0.175]
apr = scores(df, mini[0]-0.065)
apr2 = scores(df, mini[1]-0.045)
apr3 = scores(df, mini[2]-0.047)
apr4 = scores(df, 0.132)
df1=pd.merge(df1, dft[['CustomerID', 'season', 'StockCode']], on=['CustomerID', 'season'], how='inner')
df1=df1.drop_duplicates(keep='last')
df1['Items']=df1[[]].values.tolist()
####for customers who bought more than 550 items, around 390 items are recommended, items number varies by seasons see function#####
df1=recom_seasons(df1).split_seasons(390, pop_act)
####for customers who bought 550 >items>500, around 330 active customer popular items are recommended, items number varies by seasons see function#####
df1=recom_seasons(df1).group_recom_seasons(330,  550, pop_act)
####for customers who bought 500 >items>450, around 320 active customer popular items are recommended by seasons, items number varies by seasons see function#####
df1=recom_seasons(df1).group_recom_seasons(320,  500, pop_act)
####for customers who bought 450 >items>400, around 270 items are recommended by seasons, items number varies by seasons see function#####
df1=recom_seasons(df1).group_recom_seasons(270,  450, pop_seasons)
####for customers who bought 400 >items>350, around 330 items are recommended by seasons, items number varies by seasons see function#####
df1=recom_seasons(df1).group_recom_seasons(230,  400, pop_seasons)
####for customers who bought 350 >items>300, around 230 items are recommended by seasons, plus some apriori items are added#####
df1=group_recom_apr(df1, apr4, 350, 230, mini[3], 0.132, 'seasons')#group_recom_seasons(df1, 230, 350, pop_winter, pop_spring, pop_summer, pop_fall)
####for customers who bought 300 > items > 250, items are recommended based on apriori score#####
df1=group_recom_apr(df1, apr, 300, 0, mini[0], mini[0]-0.065, 'apriori')
####some apriori score recommendation for customers who bought 250>items>200
##df1=group_recom_apr(df1, apr2, 250, 0, mini[1], mini[1]-0.045, 'apriori')
assoc, apr1_up, apr_item2, apr_item1 = association(apr2, mini[1], mini[1]-0.045)

l1 = assoc['item1'].tolist()
l2=assoc['item2'].tolist()

dftemp = df1[(df1['item_count']<250)]
dftemp['Items']=dftemp[[]].values.tolist()
for i in range(len(l1)):
    dftemp1=dftemp[dftemp['StockCode'].isin([l1[i]])]
    it1 = l2[i]
    dftemp1['Items'] = dftemp1['Items'].apply(lambda x: x + it1)
    dftemp[dftemp['StockCode'].isin([l1[i]])]=dftemp1
dftemp['Items']=dftemp['Items'].apply(lambda x: list(set(list(x)+list(apr1_up))))#pop_cust[0:250]
df1.loc[df1['item_count']<250]=dftemp
#####Here are recomendations by item popularity over all seasons #########
x1=[170,150,120,100,70]
x2=[200,150,120,100,70]
for i in zip(x1, x2):
    df1=recom_seasons(df1).group_recom(pop_cust[0:i[0]], i[1])

####apri score recomendations are added####################
assoc, apr1_up, apr_item2, apr_item1 = association(apr3, mini[2], mini[2]-0.047)
l1 = assoc['item1'].tolist()
l2=assoc['item2'].tolist()
for i in range(len(l1)):
    dftemp=df1.loc[df1['StockCode'].isin([l1[i]])]
    it1 = l2[i]
    dftemp['Items'] = dftemp['Items'].apply(lambda x: x + it1)
    df1.loc[df1['StockCode'].isin([l1[i]])]=dftemp
##############low activity groups, popular items are recommended##############
x1=[60,50,40,20,10]
x2=[60,50,30,20,10]
for i in zip(x1, x2):
    df1=recom_seasons(df1).group_recom(pop_cust[0:i[0]], i[1])
###########item which goes with other item, found during data exploration###############
i1='22748P'
i2='22745M'
dfn=df1[(df1['StockCode'] == i1) & (df1['StockCode'] != i2)]
dfn[['Items']] = dfn[['Items']].apply(lambda x: x+[i2])
df1[(df1['StockCode'] == i1) & (df1['StockCode'] !=i2)]=dfn
###remove items which are already bought#############
df1[['Items']]=df1[['Items', 'StockCode']].apply(lambda x: remove_extra(x['Items'], x['StockCode']), axis=1)
df1=df1[['CustomerID', 'Items']]
df1 = df1[['CustomerID', 'Items']].groupby('CustomerID', as_index=False).agg(list)
df1['Items']=df1['Items'].apply(lambda x: list(set([item for sublist in x for item in sublist])))    
#print(df1)
df1['Items']=df1['Items'].apply(lambda x: ' '.join(x))
df1=df1.drop_duplicates(keep='last')
df1['Items']=df1['Items'].apply(lambda x: x.split(' '))
df1.to_csv('result.csv', index=False)
print(df1)
#df2 = recommendation_ff()
#print(df1.equals(df2))
del df
del dftr
del dft
del df1
