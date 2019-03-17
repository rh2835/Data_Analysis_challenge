#import data
import pandas as pd
import re
from nltk import tokenize
import numpy as np

filename = 'Diet Coke Raw Data.csv'
df = pd.read_csv(filename, skiprows = [0,1,2,3], sep='`')

#section 1: metrics
df['month'] = df[df['Date'].notnull()]['Date'].apply(lambda x: x[5:7])
df1 = df[(df['month'] != '') & (df['month'] != ' n') & (df['month'] != '8')]

csv1 = df1.groupby('month').count()
csv1 = csv1[['Full_Text']]

#1.1.2: Identify total engagement for each month
#change string type to float
def f(input_list):
        out_list = []
        l = len(input_list)
        for i in range(l):
            if isinstance(input_list[i],str):
                out_list.append(0.0)
            else:
                out_list.append(input_list[i])
        return out_list

#filter nan with 0 in 'Blog_Comments'
input_list1 = df['Blog_Comments'].fillna(0.0)
#change string in 'Blog_Comments' to float 0
list_blog_comments = f(input_list1)
df['Blog_Comments'] = list_blog_comments

#filter nan with 0 in 'Facebook_Comments'
input_list2 = df['Facebook_Comments'].fillna(0.0)
#change string in 'Facebook_Comments' to float 0
list_face_comments = f(input_list2)
df['Facebook_Comments'] = list_face_comments

#filter nan with 0 in 'Facebook_Likes'
input_list3 = df['Facebook_Likes'].fillna(0.0)
#change string in 'Facebook_Likes' to float 0
list_face_likes = f(input_list3)
df['Facebook_Likes'] = list_face_likes

#filter nan with 0 in 'Instagram_Comments'
input_list4 = df['Instagram_Comments'].fillna(0.0)
#change string in 'Instagram_Comments' to float 0
list_inst_comments = f(input_list4)
df['Instagram_Comments'] = list_inst_comments

#filter nan with 0 in 'Instagram_Likes'
input_list5 = df['Instagram_Likes'].fillna(0.0)
#change string in 'Instagram_Likes' to float 0
list_inst_likes = f(input_list5)
df['Instagram_Likes'] = list_inst_likes

#filter nan with 0 in 'Twitter_Retweets'
input_list6 = df['Twitter_Retweets'].fillna(0.0)
#change string in 'Twitter_Retweets' to float 0
list_twit_retweet = f(input_list6)
df['Twitter_Retweets'] = list_twit_retweet

df['Social_Engagement'] = df['Blog_Comments']+df['Facebook_Comments']+df['Facebook_Likes']+df['Instagram_Comments']+df['Instagram_Likes']+df['Twitter_Retweets']

df2 = df[(df['month'] != '') & (df['month'] != ' n') & (df['month'] != '8')]
csv2 = df2.groupby('month').sum()
csv2 = csv2[['Social_Engagement']]

#1.2.1: Filter out unwanted mentions
df3 = df[(df['Page_Type'] == 'instagram') | (df['Page_Type'] == 'twitter') | (df['Page_Type'] == 'facebook') | (df['Page_Type'] == 'forum') | (df['Page_Type'] == 'blog') | (df['Page_Type'] == 'Twitter') | (df['Page_Type'] == 'news') | (df['Page_Type'] == 'Instagram') | (df['Page_Type'] == 'Forum')]

#Group related mentions into 3 categories
dic = {'instagram': 'social',
       'twitter': 'social',
       'facebook': 'social',
       'forum': 'blog & forum',
       'blog': 'blog & forum',
       'Twitter': 'social',
       'news': 'news',
       'Instagram': 'social',
       'Forum': 'blog & forum'
      }

df3['mention_category'] = df3['Page_Type'].map(dic)
df3 = df3[['Sentiment','mention_category']]
df3 = df3[(df3['Sentiment']=='1.0') | (df3['Sentiment']=='0.5') | (df3['Sentiment']=='0.0') | (df3['Sentiment']==1.0) | (df3['Sentiment']==0.5) | (df3['Sentiment']==0.0)]
df3['Sentiment'] = [float(x) for x in df3['Sentiment']]

#Calculate sum of sentiment for each category
social_df3 = df3[df3['mention_category'] == 'social']
csv3 = social_df3.groupby('Sentiment').count()
csv3 = csv3.rename(columns = {"mention_category": "social"})

news_df3 = df3[df3['mention_category'] == 'news']
csv4 = news_df3.groupby('Sentiment').count()
csv4 = csv4.rename(columns = {"mention_category": "news"})

bf_df3 = df3[df3['mention_category'] == 'blog & forum']
csv5 = bf_df3.groupby('Sentiment').count()
csv5 = csv5.rename(columns = {"mention_category": "blog & forum"})

#As calculated above, social has the highest sentiment. The total number of positive mentions of social is 158470.

#1.3: Find the top 10 authors with the highest followers
#sort instagram followers
input_list7 = df['Instagram_Followers'].fillna(0.0)
df['Instagram_Followers'] = f(input_list7)
inst_sort_df = df.sort_values(by='Instagram_Followers', ascending=False)
csv6 = inst_sort_df.head(10) 
csv6 = csv6[['Author','Instagram_Followers', 'Url']]

#sort twitter followers
input_list8 = df['Twitter_Followers'].fillna(0.0)
df['Twitter_Followers'] = f(input_list8)
twit_sort_df = df.sort_values(by='Twitter_Followers', ascending=False)
csv7 = twit_sort_df.head(10)
csv7 = csv7[['Author','Twitter_Followers', 'Url']]

#1.4: List all the news websites
news_df = df[df['Page_Type'] == 'news']
i = pd.Series(range(0,70))
news_df = news_df.set_index([i])
l1 = news_df['Url'].apply(lambda x: x.split('/')[2])
l2 = l1.apply(lambda x: x.split('.'))

websites = []
for i in range(70):
    if len(l2[i]) == 2:
        websites.append(l2[i][0])
    else:
        websites.append(l2[i][1])


#Section 2: Topic Analysis
df3 = df[(df['Page_Type'] == 'instagram') | (df['Page_Type'] == 'twitter') | (df['Page_Type'] == 'facebook') | (df['Page_Type'] == 'forum') | (df['Page_Type'] == 'blog') | (df['Page_Type'] == 'Twitter') | (df['Page_Type'] == 'news') | (df['Page_Type'] == 'Instagram') | (df['Page_Type'] == 'Forum')]
dic = {'instagram': 'social',
       'twitter': 'social',
       'facebook': 'social',
       'forum': 'blog & forum',
       'blog': 'blog & forum',
       'Twitter': 'social',
       'news': 'news',
       'Instagram': 'social',
       'Forum': 'blog & forum'
      }
df3['mention_category'] = df3['Page_Type'].map(dic)

#2.1: Tag all of the official mention to corresponding brands
#convert all 'Full_Text' type to lower alphabet strings
df3['Full_Text'] = df3['Full_Text'].fillna('0')
texts = df3['Full_Text'].apply(lambda x: str(x).lower())

lists = texts.tolist()
l = len(lists)
official = []

for i in range(l):
    if ('dietcoke' in lists[i])|('dietcokeus' in lists[i]):
        official.append('Diet Coke')
    elif ('lacroixwater' in lists[i])|('lacroix' in lists[i]):
        official.append('Lacroix')
    elif ('vitacoco' in lists[i])|('vitacocous' in lists[i]):
        official.append('Vita Coco')
    elif ('warbyparker' in lists[i]):
        official.append('Warby Parker')
    elif 'everlane' in lists[i]:
        official.append('Everlane')
    elif 'venmo' in lists[i]:
        official.append('Venmo')
    elif 'glossier' in lists[i]:
        official.append('Glossier')
    elif 'casper' in lists[i]:
        official.append('Casper')
    elif 'dollarshaveclub' in lists[i]:
        official.append('Dollar Shave Club')
    elif 'birchbox' in lists[i]:
        official.append('Birchbox')
    elif 'bonobos' in lists[i]:
        official.append('Bonobos')
    else:
        official.append('UGC')
        
df3['Official'] = official

#2.2: Tag all 'UGC' of 'social' into corresponding topics
def excit(text):
    return ('excited' in text)|('exciting' in text)

def surprise(text):
    return ('surprised' in text)|('amazing' in text)|('gosh' in text)|('omg' in text)|('shocked' in text)|('impress' in text)|('thrill' in text)|('magic' in text)

def pleasant(text):
    return ('pleasant' in text)|('joyful' in text)|('amused' in text)|('glad' in text)|('delightful' in text)|('happy' in text)|('happiness' in text)

def refresh(text):
    return ('refreshed' in text)|('rejuvenated' in text)|('relive' in text)|('revived' in text)|bool(re.search(r'new\s?life', text))|('nourish' in text)|('refreshing' in text)

def energe(text):
    return bool('energetic' in text)|('lively' in text)|('powerful' in text)|('spirited' in text)|('vigorous' in text)|('vibrant' in text)|('uplift' in text)|('pumped' in text)|('energized' in text)

def rest(text):
    return ('restful' in text)|('peaceful' in text)|('calm' in text)|('relaxed' in text)|('unwind' in text)|('destress' in text)|('meditat' in text)|('sooth' in text)|('comfort' in text)|('tranquil' in text)|('enjoyed' in text)|('spaciou' in text)|('solitud' in text)|('retreat' in text)|('rested' in text)

def thank(text):
    return ('thankful' in text)|bool(re.search(r'thank\s?god', text))|('thanks' in text)|bool(re.search(r'thank\s?you', text))

def angry(text):
    return ('angry' in text)|('infuriated' in text)|('wrath' in text)|('unhappy' in text)|('horrible' in text)|('hate' in text)|('stupid' in text)|('scream' in text)|('irritat' in text)|('dread' in text)|('pissing' in text)|('grumbl' in text)|('mad' in text)|('pisses' in text)|('pissed' in text)

def annoy(text):
    return ('annoyed' in text)|('pique' in text)|('uncomfort' in text)|('bother' in text)|('miffed' in text)|('irke' in text)

def anxiety(text):
    return ('anxious' in text)|('embarrass' in text)|('anxiety' in text)|('afraid' in text)|('concerned' in text)|('wreck' in text)|('jumpy' in text)|('bugged' in text)|('disturbed' in text)|('fretful' in text)|('worri' in text)|('depress' in text)|('worry' in text)

def frust(text):
    return ('frustrated' in text)|('upset' in text)|('discouraged' in text)|bool(re.search(r'fouled\s?up',text))|bool(re.search(r'hung\s?up\s?on',text))|('resentful' in text)|bool(re.search(r'up\s?the\s?wall',text))|('ungratified' in text)

def power(text):
    return ('powerless' in text)|('weak' in text)|('overwhelmed' in text)|('insecur' in text)|('frighten' in text)|('miser' in text)|('resent' in text)|('lonel' in text)|('dreary' in text)|('fatigued' in text)

def stress(text):
    return ('stressful' in text)|('pressure' in text)|('fatigu' in text)|('distract' in text)|('cortisol' in text)|('burnout' in text)|('meltdown' in text)|('insomnia' in text)|('procrastin' in text)|('mental' in text)|('headach' in text)|('drows' in text)|('nervou' in text)|('chore' in text)|('sluggish' in text)|('exhaust' in text)

def sad(text):
    return ('sore' in text)|('sad' in text)|bool(re.search(r'mood\s?dropped',text))


def topic(text):
    topics = []
    
    if excit(text):
        topics.append('Excitement')
    if surprise(text):
        topics.append('Surprise')
    if pleasant(text):
        topics.append('Pleasant')
    if refresh(text):
        topics.append('Refreshed')
    if energe(text):
        topics.append('Energetic')
    if rest(text):
        topics.append('Restful')
    if thank(text):
        topics.append('Thankful')
    if angry(text):
        topics.append('Angry')
    if annoy(text):
        topics.append('Annoy')
    if anxiety(text):
        topics.append('Anxiety')
    if frust(text):
        topics.append('Frustration')
    if power(text):
        topics.append('Powerless')
    if stress(text):
        topics.append('Stressful')
    if sad(text):
        topics.append('Sad')
    return topics

#Extract mentions that come from 'UGC' and 'social'
df4 = df3[df3['mention_category'] == 'social']
df4['Full_Text'] = df4['Full_Text'].apply(lambda x: x.lower())
df4.reset_index(inplace=True)

#Identify topic to each 'Full_Text'
topic_list = []
l = len(df4['Full_Text'])

for i in range(l):
    s = df4['Full_Text'][i]
    topic_list.append(topic(s))

df4['Topics'] = topic_list

#2.3: Find number of mentions, total engagement, positive sentiment percentage for each topic
df4['Excitement'] = df4['Topics'].apply(lambda x: 'Excitement' in x)
df4['Surprise'] = df4['Topics'].apply(lambda x: 'Surprise' in x)
df4['Pleasant'] = df4['Topics'].apply(lambda x: 'Pleasant' in x)
df4['Refreshed'] = df4['Topics'].apply(lambda x: 'Refreshed' in x)
df4['Energetic'] = df4['Topics'].apply(lambda x: 'Energetic' in x)
df4['Restful'] = df4['Topics'].apply(lambda x: 'Restful' in x)
df4['Thankful'] = df4['Topics'].apply(lambda x: 'Thankful' in x)
df4['Angry'] = df4['Topics'].apply(lambda x: 'Angry' in x)
df4['Annoy'] = df4['Topics'].apply(lambda x: 'Annoy' in x)
df4['Anxiety'] = df4['Topics'].apply(lambda x: 'Anxiety' in x)
df4['Frustration'] = df4['Topics'].apply(lambda x: 'Frustration' in x)
df4['Powerless'] = df4['Topics'].apply(lambda x: 'Powerless' in x)
df4['Stressful'] = df4['Topics'].apply(lambda x: 'Stressful' in x)
df4['Sad'] = df4['Topics'].apply(lambda x: 'Sad' in x)


def f(topic):
    r = []
    r1 = len(df4[df4[topic] == True]['Full_Text'])
    r2 = sum(df4[df4[topic] == True]['Social_Engagement'])
    r3 = sum(df4[df4[topic] == True]['Sentiment']==1)/len(df4[df4[topic] == True]['Sentiment'])
    r.append(r1)
    r.append(r2)
    r.append(r3)
    return r

d = {'Excitement': f('Excitement'), 
     'Surprise': f('Surprise'),
    'Pleasant': f('Pleasant'),
    'Refreshed': f('Refreshed'),
    'Energetic': f('Energetic'),
    'Restful': f('Restful'),
    'Thankful': f('Thankful'),
    'Angry': f('Angry'),
    'Annoy': f('Annoy'),
    'Anxiety': f('Anxiety'),
    'Frustration': f('Frustration'),
    'Powerless': f('Powerless'),
    'Stressful': f('Stressful'),
    'Sad': f('Sad')}
results = pd.DataFrame(data=d)
results.rename({0: 'number of mentions', 1: 'total engagement', 2: 'positive sentiment percentage'})


#Section 3: Longform Analysis
#3.1: Classifying topics for longform articles and finding sentiment score
df5 = df3[(df3['mention_category'] != ('social'))]
df5['Full_Text'] = df5['Full_Text'].apply(lambda x: x.lower())
df5.reset_index(inplace=True)

#Identify topics for each longform article
longtopic_list = []
l = len(df5['Full_Text'])

for i in range(l):
    s = df5['Full_Text'][i]
    longtopic_list.append(topic(s))

df5['Topics'] = longtopic_list

#Find sentiment score for every topic in each longform article
df5['Excitement'] = df5['Topics'].apply(lambda x: 'Excitement' in x)
df5['Surprise'] = df5['Topics'].apply(lambda x: 'Surprise' in x)
df5['Pleasant'] = df5['Topics'].apply(lambda x: 'Pleasant' in x)
df5['Refreshed'] = df5['Topics'].apply(lambda x: 'Refreshed' in x)
df5['Energetic'] = df5['Topics'].apply(lambda x: 'Energetic' in x)
df5['Restful'] = df5['Topics'].apply(lambda x: 'Restful' in x)
df5['Thankful'] = df5['Topics'].apply(lambda x: 'Thankful' in x)
df5['Angry'] = df5['Topics'].apply(lambda x: 'Angry' in x)
df5['Annoy'] = df5['Topics'].apply(lambda x: 'Annoy' in x)
df5['Anxiety'] = df5['Topics'].apply(lambda x: 'Anxiety' in x)
df5['Frustration'] = df5['Topics'].apply(lambda x: 'Frustration' in x)
df5['Powerless'] = df5['Topics'].apply(lambda x: 'Powerless' in x)
df5['Stressful'] = df5['Topics'].apply(lambda x: 'Stressful' in x)
df5['Sad'] = df5['Topics'].apply(lambda x: 'Sad' in x)

df6 = df5[['Full_Text','Sentiment','Excitement','Surprise','Pleasant','Refreshed','Energetic','Restful','Thankful','Angry','Annoy','Anxiety','Frustration','Powerless','Stressful','Sad']]

#3.2: Attach key content
#find non-empty topic full_texts' index
def key_content_ind(longtopic_list):
    ind = []
    
    l = len(longtopic_list)
    for i in range(l):
        if longtopic_list[i] != []:
            ind.append(i)
    return ind
ind = key_content_ind(longtopic_list)

#identify each sentence contains topic or not
def contain_topic(text):
    return (excit(text))|(surprise(text))|(pleasant(text))|(refresh(text))|(energe(text))|(rest(text))|(thank(text))|(angry(text))|(annoy(text))|(anxiety(text))|(frust(text))|(power(text))|(stress(text))|(sad(text))

#return non_repeat and individual sentences
def non_repeat(t):
    #split text into sentences
    sentences = tokenize.sent_tokenize(t)
    l = len(sentences)
    
    res = []
    res.append(sentences[0])
    
    for i in range(l):
        if sentences[i] not in res:
            res.append(sentences[i])
        else:
            continue
    return res

#return key_content index
def key_content_ind(l):
    results = []
    compound = []
    le = len(l)
    for i in range(le):
        string = l[i]
        compound = []
        if (contain_topic(string)):
                if(i-2 >= 0):
                    compound.append(i-2)
                if(i-1 >= 0):
                    compound.append(i-1)
                compound.append(i)
                if(i+1 < le):
                    compound.append(i+1)
                if(i+2 < le):
                    compound.append(i+2) 
        results.append(compound)        
    return results

#return key content for each full text
def key_content(text):
    key_content = []
    sentence_list = non_repeat(text)
    idx = key_content_ind(sentence_list)
    index = [x for x in idx if x!=[]][0]
    for x in index:
        key_content.append(sentence_list[x])
    return key_content

res = []
for i in range(len(longtopic_list)):
    
    if longtopic_list[i] == []:
        res.append([])
    else:
        res.append(key_content((df5['Full_Text'][i])))      
df5['key_content'] = res

#3.3: calculate number of topics matched for each longform article
df5['number_of_topics'] = df5['Topics'].apply(lambda x: len(x))

#3.4: Sentiment analysis on key content
#Use code from: https://github.com/youhealthy/vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
def sentiment(content):
    vs_list = []
    
    for sentence in content:
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(sentence)
        vs_list.append(vs)
    return vs_list

sentiment_list = df5['key_content'].apply(lambda x: sentiment(x))
df5['Sentiment_analysis'] = sentiment_list


#Section 4: Dashboard
#4.1: Line graph to indicate weekly change of volume of mentions
df1['Date'] = df1['Date'].apply(lambda x: pd.to_datetime(x))
df6 = df1[df1['Date'].notnull()]
df6['Week'] = df6['Date'].apply(lambda x: x.isocalendar()[1])
df6['Year'] = df6['Date'].apply(lambda x: x.isocalendar()[0])

df_1917 = df6[df6['Year']==1917]
df_2017 = df6[df6['Year']==2017]
df_2018 = df6[df6['Year']==2018]

res1917 = df_1917.groupby('Week').count()
volume1917 = res1917[['Full_Text']]
res2017 = df_2017.groupby('Week').count()
volume2017 = res2017[['Full_Text']]
res2018 = df_2018.groupby('Week').count()
volume2018 = res2018[['Full_Text']]
volume = volume1917.append(volume2017)
volume = volume.append(volume2018)
volume_df = volume.reset_index(drop=True)

import matplotlib.pyplot as plt
plt.plot(volume_df)
plt.title('weekly volume of mentions change')

#4.2: List top 10 websites by volume of mentions for News and Blog & Forum
news_df['websites'] = websites
res_df = news_df.groupby('websites').count()
res_df = res_df[['Full_Text']]
sort_websites = res_df.sort_values(by=['Full_Text'],ascending=False)
res = sort_websites[0:10]
res = res.rename(columns = {"Full_Text": "news"})

objects = list(res.index)
y_pos = np.arange(10)
performance = res['Full_Text']
plt.figure(figsize=(20,10))
plt.bar(y_pos, performance, alpha=0.5)
plt.xticks(y_pos, objects)
plt.title('top 10 websites by volume of mentions for News')

type_df = df[(df['Page_Type'] == 'forum')|(df['Page_Type'] == 'Forum')|(df['Page_Type'] == 'blog')]

i = pd.Series(range(0,41858))
type_df = type_df.set_index([i])
l1 = type_df['Url'].apply(lambda x: x.split('/')[2])
l2 = l1.apply(lambda x: x.split('.'))

websites2 = []
for i in range(41858):
    if len(l2[i]) == 2:
        websites2.append(l2[i][0])
    else:
        websites2.append(l2[i][1])

type_df['websites'] = websites2
res_df2 = type_df.groupby('websites').count()
res_df2 = res_df2[['Full_Text']]
sort_websites2 = res_df2.sort_values(by=['Full_Text'],ascending=False)
res2 = sort_websites2[0:10]
res2 = res2.rename(columns = {"Full_Text": "blog & forum"})

objects = list(res2.index)
y_pos = np.arange(10)
performance = res2['Full_Text']
plt.figure(figsize=(20,10))
plt.bar(y_pos, performance, alpha=0.5)
plt.xticks(y_pos, objects)
plt.title('top 10 websites by volume of mentions for Blog & Forum')

#4.3: Engagement weekly change in the Social category
df7 = df3[df3['mention_category']=='social']
df7['Date'] = df7['Date'].apply(lambda x: pd.to_datetime(x))
df7 = df7[df7['Date'].notnull()]
df7['Week'] = df7['Date'].apply(lambda x: x.isocalendar()[1])
df7['Year'] = df7['Date'].apply(lambda x: x.isocalendar()[0])
df7 = df7[(df7['Sentiment'] == '1.0')|(df7['Sentiment'] == '0.5')|(df7['Sentiment'] == '0.0')|(df7['Sentiment'] == 1.0)|(df7['Sentiment'] == 0.5)|(df7['Sentiment'] == 0)]
df7['Sentiment'] = df7['Sentiment'].apply(lambda x: float(x))

df_1917 = df7[df7['Year']==1917]
df_2017 = df7[df7['Year']==2017]
df_2018 = df7[df7['Year']==2018]

def cal(week, df):
    value_list = []
    sub_df = df[df['Week']==week]
    value_list.append(len(sub_df[sub_df['Sentiment']==1])/len(sub_df['Sentiment']))
    value_list.append(sum(sub_df['Social_Engagement']))
    return value_list

def pos(week_list, df):
    pos = []
    for i in range(len(week_list)):
        values = cal(week_list[i], df)
        pos.append(values[0])
    return pos

def engage(week_list, df):
    engage = []
    for i in range(len(week_list)):
        values = cal(week_list[i], df)
        engage.append(values[1])
    return engage    

week1917 = df_1917['Week'].unique()
pos1917 = pos(week1917, df_1917)
engage1917 = engage(week1917, df_1917)

week2017 = df_2017['Week'].unique()
pos2017 = pos(week2017, df_2017)
engage2017 = engage(week2017, df_2017)

week2018 = df_2018['Week'].unique()
pos2018 = pos(week2018, df_2018)
engage2018 = engage(week2018, df_2018)

data1917 = {'week': week1917, 'pos': pos1917, 'engage': engage1917}
combine1917 = pd.DataFrame(data1917)
combine1917 = combine1917.sort_values(by='week')

data2017 = {'week': week2017, 'pos': pos2017, 'engage': engage2017}
combine2017 = pd.DataFrame(data2017)
combine2017 = combine2017.sort_values(by='week')

data2018 = {'week': week2018, 'pos': pos2018, 'engage': engage2018}
combine2018 = pd.DataFrame(data2018)
combine2018 = combine2018.sort_values(by='week')

pos = combine1917['pos'].append(combine2017['pos'])
pos = pos.append(combine2018['pos'])

engage = combine1917['engage'].append(combine2017['engage'])
engage = engage.append(combine2018['engage'])

data = {'pos': pos, 'engage': engage}
res_df = pd.DataFrame(data)
res_df = res_df.reset_index(drop = True)

res_df = res_df.rename(columns = {"pos": "percentage of positive mentions", "engage": "engagement"})

plt.figure(figsize=(20,10))
plt.scatter(list(res_df.index), res_df["percentage of positive mentions"], s=res_df["engagement"]/100)
plt.title('Engagement weekly change in the Social category')