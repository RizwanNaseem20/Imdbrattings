import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Reading the Data 

movie_df=pd.read_csv("F:/MoviesuccesPrediction/Sample.csv " ,encoding='latin1')
#Displaying the first 10 records

movie_df.head(10)

movie_df.drop('movie_imdb_link', axis=1, inplace=True)
#Removing the color section as most of the movies is colored

movie_df["color"].value_counts()

movie_df.drop('color',axis=1,inplace=True)
#Checking for the columns present in the datset
movie_df.columns

# We can remove the null values from the dataset where the count is less . so that we don't loose much data 

movie_df.dropna(axis=0,subset=['director_name', 'num_critic_for_reviews','duration','director_facebook_likes','actor_3_facebook_likes','actor_2_name','actor_1_facebook_likes','actor_1_name','actor_3_name','facenumber_in_poster','num_user_for_reviews','language','country','actor_2_facebook_likes','plot_keywords'],inplace=True)


#Replacing the content rating with Value R as it has highest frequency

movie_df["content_rating"].fillna("R", inplace = True) 
#Replacing the aspect_ratio with the median of the value as the graph is right skewed 

movie_df["aspect_ratio"].fillna(movie_df["aspect_ratio"].median(),inplace=True)
#We need to replace the value in budget with the median of the value

movie_df["budget"].fillna(movie_df["budget"].median(),inplace=True)
# We need to replace the value in gross with the median of the value 

movie_df['gross'].fillna(movie_df['gross'].median(),inplace=True)

#Removing the duplicate values in the datset

movie_df.drop_duplicates(inplace=True)

#Most of the values for the languages is english we can drop the english column

movie_df.drop('language',axis=1,inplace=True)
#Creating a new column to check the net profit made by the company (Gross-Budget) 

movie_df["Profit"]=movie_df['budget'].sub(movie_df['gross'], axis = 0) 



movie_df['Profit_Percentage']=(movie_df["Profit"]/movie_df["gross"])*100
movie_df



value_counts=movie_df["country"].value_counts()
##get top 2 values of index
vals = value_counts[:2].index
print (vals)
movie_df['country'] = movie_df.country.where(movie_df.country.isin(vals), 'other')

#Successfully divided the country into three catogories 
movie_df["country"].value_counts()

print ()
print ()
print ()
print ("--------------Top 20 movies based on the profit they made-------------------------------")
plt.figure(figsize=(10,8))
movie_df= movie_df.sort_values(by ='Profit' , ascending=False)
movie_df_new=movie_df.head(20)
ax=sns.pointplot(movie_df_new['Profit'], movie_df_new['budget'], hue=movie_df_new['movie_title'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


#Removing the director name column

movie_df.drop('director_name', axis=1, inplace=True)
#Removing the actor1 ,actor 2 and actor 3 names 

movie_df.drop('actor_1_name',axis=1,inplace=True)
movie_df.drop('actor_2_name',axis=1,inplace=True)
movie_df.drop('actor_3_name',axis=1,inplace=True)
#Dropping the movie title 

movie_df.drop('movie_title',axis=1,inplace=True)
# Dropping the plot keywords
movie_df.drop('plot_keywords',axis=1,inplace=True)
#Value count of genres

movie_df['genres'].value_counts()

#Most of the values are equally distributed in genres column ,so we can remove the genres column

movie_df.drop('genres',axis=1,inplace =True)

# Dropiing the profit column from the dataset
movie_df.drop('Profit',axis=1,inplace=True)

#Dropping the profit percentage column from the dataset

movie_df.drop('Profit_Percentage',axis=1,inplace=True)


print ()
print ()
print ()
print ("--------------------Correlation with heat map-------------------------")


# Correlation with heat map
import matplotlib.pyplot as plt
import seaborn as sns
corr = movie_df.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(13,7))
# create a mask so we only see the correlation values once
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)







#Adding the facebook likes of actor 2 and actor 3 together 
movie_df['Other_actor_facebbok_likes']=movie_df["actor_2_facebook_likes"] + movie_df['actor_3_facebook_likes']
#Dropping the actor 2 and actor 3 facebook likes columns as they have been added together 

movie_df.drop('actor_2_facebook_likes',axis=1,inplace=True)
movie_df.drop('actor_3_facebook_likes',axis=1,inplace=True)
movie_df.drop('cast_total_facebook_likes',axis=1,inplace=True)
#Ratio of the ratio of num_user_for_reviews and num_critic_for_reviews.

movie_df['critic_review_ratio']=movie_df['num_critic_for_reviews']/movie_df['num_user_for_reviews']
#Dropping the num_critic_for_review

movie_df.drop('num_critic_for_reviews',axis=1,inplace=True)
movie_df.drop('num_user_for_reviews',axis=1,inplace=True)


print ()
print ()
print ()
print ("-------------------New Correlation matrix shown in the figure -------------------------")

import matplotlib.pyplot as plt
import seaborn as sns
corr = movie_df.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(13,7))
# create a mask so we only see the correlation values once
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
# create a mask so we only see the correlation values once


# We need to categorize the imdb values in the range of 0-4,4-6,6-8 and 8-10 to mark them as the bad,average,good and excellent movies respectively

movie_df["imdb_binned_score"]=pd.cut(movie_df['Result'], bins=[0,4,6,8,10], right=True, labels=False)+1
#Dropping the imdb_score column as it is being replaced with the imdb_binned_score values 
movie_df.drop('Result',axis=1,inplace=True)
movie_df.head(5)
