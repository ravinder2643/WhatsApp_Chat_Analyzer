from collections import Counter
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from urlextract import URLExtract
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import requests
import emoji
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import seaborn as sns
from datetime import datetime
import google.generativeai as palm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation




import streamlit as st
def stats(user_type,df ):
    if user_type is not "Overall":
        df= df[df['user']==user_type]
    words=[]
    for messages in df['message']:
        words.extend(messages.split())
        
    media = df[df['message'].str.contains('<Media omitted>', case=False, na=False)]
    
    urls=[]
    extractor = URLExtract()

    for messages in df['message']:
        urls.extend (extractor.find_urls(messages) )

        
    return len(df), len(words), len(media), len(urls)

def most_busy_user(df):
    
    
    
    cf = df
    cf = cf[~cf['user'].str.contains('added')]
    cf = cf[~cf['user'].str.contains('left')]
    cf = cf[~cf['user'].str.contains('security')]
    cf = cf[~cf['user'].str.contains('changed')]
    cf = cf[~cf['user'].str.contains('removed')]
    cf = cf[~cf['user'].str.contains('deleted')]
    cf = cf[~cf['user'].str.contains('group-notification')]
    cf = cf[~cf['user'].str.contains('joined')]
    cf = cf[~cf['user'].str.contains('created')]
    x = cf['user'].value_counts().head()


    y=pd.DataFrame(round(cf['user'].value_counts()/len(cf)*100,2)).reset_index().rename(columns={'user':'name','count':'percent'})        
    return x,y
    


def get_wordcloud(user_type,df):
    if user_type is not "Overall":
        df= df[df['user']==user_type]
    cf=df
    keywords = ['added', 'left', 'security', 'changed', 'removed', 'deleted', 'group-notification']
    cf = cf[~cf['user'].str.contains('|'.join(keywords))]
    cf = cf[~cf['message'].str.contains('<Media omitted>')]
    f=open('stop_hinglish.txt','r')
    stop_hinglish=f.read()
    words=[]
    for message in cf['message']:
        for word in message.lower().split():
            if word not in stop_hinglish:
                words.append(word)
    from collections import Counter
    f.close()
    y=pd.DataFrame(Counter(words).most_common(len(Counter(words)))).reset_index().drop(columns=['index']).rename(columns={0:'words',1:'freq'})
    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    df_wc= wc.generate(y['words'].str.cat(sep=' '))
    return df_wc

def top_com_words(user_type,df):
    if user_type is not "Overall":
        df= df[df['user']==user_type]
    cf=df
    keywords = ['added', 'left', 'security', 'changed', 'removed', 'deleted', 'group-notification']
    cf = cf[~cf['user'].str.contains('|'.join(keywords))]
    cf = cf[~cf['message'].str.contains('<Media omitted>')]
    f=open('stop_hinglish.txt','r')
    stop_hinglish=f.read()
    words=[]
    for message in cf['message']:
        for word in message.lower().split():
            if word not in stop_hinglish:
                words.append(word)
    from collections import Counter
    f.close()
    y=pd.DataFrame(Counter(words).most_common(20)).reset_index().drop(columns=['index']).rename(columns={0:'words',1:'frequency'})
    return y

def fetch_message(df,selected_user):
    df= df[df['user']==selected_user]
    return df


def top_emoji(user_type,df ):
    if user_type is not "Overall":
        df= df[df['user']==user_type]
        
    emojis=[]
    for message in df["message"]:
        emojis.extend(emoji.emoji_list(message))
    emojis = [entry['emoji'] for entry in emojis]
    y=pd.DataFrame(Counter(emojis).most_common(10)).rename(columns={0:'emoji',1:'frequency'})
    return y


def time_line(user_type,df ):
    if user_type is not "Overall":
        df= df[df['user']==user_type]
    cf = df
    cf = g(cf.copy())
    timeline=cf.groupby(['year','month_number','month']).count()['message'].reset_index()
    time=[]
    for i in range(len(timeline)):
        time.append(timeline['month'][i]+' - '+ str(timeline['year'][i]))
    timeline['time']=time
    return timeline
    
    
def g(df):
    df['month_number'] = df['month'].map({'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12,'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12})
    return df  

def  daily_timeline(user_type,df ):
    if user_type is not "Overall":
        df= df[df['user']==user_type]
    dt= df.groupby('only_date').count()['message'].reset_index()
    return dt
def week_activity(user_type,df ):
    if user_type is not "Overall":
        df= df[df['user']==user_type]
    wt=df['dayname'].value_counts().reset_index().rename(columns={'count':'message'})
    return wt
def month_activity(user_type,df ):
    if user_type is not "Overall":
        df= df[df['user']==user_type]
    wt=df['month'].value_counts().reset_index().rename(columns={'count':'message'})
    return wt

def heat_map_data(user_type,df ):
    if user_type is not "Overall":
        df= df[df['user']==user_type]
    period=df['period'].unique()
    period = sorted(period, key=convert_to_24_hour)
    df['period'] = pd.Categorical(df['period'], categories=period, ordered=True)
    heatmap_data = df.pivot_table(index='dayname', columns='period', values='message', aggfunc='count').fillna(0)
    
    return heatmap_data

def convert_to_24_hour(time_range):
    start_time, end_time = time_range.split(' - ')
    start_dt = datetime.strptime(start_time, '%I%p')
    end_dt = datetime.strptime(end_time, '%I%p')
    
    return start_dt.strftime('%H:%M') + ' - ' + end_dt.strftime('%H:%M')   


def purify_data(df):
    df = df[~df['user'].str.contains('added')]
    df = df[~df['user'].str.contains('left')]
    df = df[~df['user'].str.contains('security')]
    df = df[~df['user'].str.contains('changed')]
    df = df[~df['user'].str.contains('removed')]
    df = df[~df['user'].str.contains('deleted')]
    df = df[~df['user'].str.contains('joined')]
    df = df[~df['user'].str.contains('created')]
    df = df[~df['message'].str.contains('<Media omitted>')]
    keywords = ['added', 'left', 'security', 'changed', 'removed', 'deleted', 'group-notification']
    df = df[~df['user'].str.contains('|'.join(keywords))]
    f=open('./stop_hinglish.txt','r')
    stop_hinglish=f.read()
    def remove_stop_words(sentence):
        return ' '.join(word for word in sentence.lower().split() if word not in stop_hinglish)

    df['message'] = df['message'].apply(remove_stop_words)
    return df

def monthly_senti_change(user_type,df ):
    if user_type is not "Overall":
        df= df[df['user']==user_type]
    df = purify_data(df)
    df['only_date'] = pd.to_datetime(df['only_date'])
    monthly_sentiment = df.groupby(df['date'].dt.to_period("M"))['sentiment_category'].value_counts(normalize=True).unstack().fillna(0)   
    return monthly_sentiment
def daily_senti_change(user_type,df ):
    if user_type is not "Overall":
        df= df[df['user']==user_type]
    df['only_date'] = pd.to_datetime(df['only_date'])
    df = purify_data(df)
    daily_sentiment = df.groupby(df['only_date'].dt.to_period("D"))['sentiment_category'].value_counts(normalize=True).unstack().reset_index().fillna(0) 
    daily_sentiment['only_date'] = daily_sentiment['only_date'].dt.to_timestamp()    
    return daily_sentiment

def monthly_emotion_change(user_type,df ):
    if user_type is not "Overall":
        df= df[df['user']==user_type]
    df = purify_data(df)
    df['only_date'] = pd.to_datetime(df['only_date'])
    monthly_emotion = df.groupby(df['only_date'].dt.to_period("M"))['emotion_label_nltk'].value_counts(normalize=True).unstack().reset_index().fillna(0)

    monthly_emotion['only_date'] = monthly_emotion['only_date'].astype(str) 
    emotion_labels = ['Anger', 'Joy', 'Neutral', 'Sadness']

    missing_columns = [label for label in emotion_labels if label not in monthly_emotion.columns.tolist()]

    if missing_columns:
        for label in missing_columns:
            monthly_emotion[label] = 0  
    return monthly_emotion

def daily_emotion_change(user_type,df ):
    if user_type is not "Overall":
        df= df[df['user']==user_type]
    df = purify_data(df)
    df['only_date'] = pd.to_datetime(df['only_date'])
    daily_emotion = df.groupby(df['only_date'].dt.to_period("D"))['emotion_label_nltk'].value_counts(normalize=True).unstack().reset_index().fillna(0)
    daily_emotion['only_date'] = daily_emotion['only_date'].dt.to_timestamp()    
    emotion_labels = ['Anger', 'Joy', 'Neutral', 'Sadness']

    missing_columns = [label for label in emotion_labels if label not in daily_emotion.columns.tolist()]

    if missing_columns:
        for label in missing_columns:
            daily_emotion[label] = 0
    return daily_emotion

def compount_sentiment_monthly(user_type,df ):
    if user_type is not "Overall":
        df= df[df['user']==user_type]
    df = purify_data(df)
    data = df.groupby('only_date')['polarity'].mean().reset_index()
    return data

def compount_emotion_monthly(user_type,df ):
    if user_type is not "Overall":
        df= df[df['user']==user_type]
    df = purify_data(df)
    data = df.groupby('only_date')['emotion_nltk'].mean().reset_index()
    return data

def subjectivity_percentage(user_type,df ):
    if user_type is not "Overall":
        df= df[df['user']==user_type]
    return df['subjectivity'].value_counts(normalize=True) * 100

def subjectivity_trend(user_type,df ):
    if user_type is not "Overall":
        df= df[df['user']==user_type]
    df['only_date'] = pd.to_datetime(df['only_date'])
    trend_df = df.groupby(df['only_date'].dt.to_period("M"))['subjectivity'].value_counts(normalize=True).unstack().reset_index().fillna(0)
    trend_df['only_date'] = trend_df['only_date'].dt.to_timestamp()

    
    return trend_df


def chat_keywords(user_type,df ):
    if user_type is not "Overall":
        df= df[df['user']==user_type]
    
    df = purify_data(df)
    palm.configure(api_key="AIzaSyBSf9YAbtiT3PBEgDsuKJlr--pb9WZj1_w") 

    messages = df['message'].astype(str)

    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(messages)

    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X)

    n_topics = 5  
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X_tfidf)

    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_keywords_idx = topic.argsort()[:-51:-1]  
        top_keywords = [feature_names[i] for i in top_keywords_idx]

        input_text = f"Generate a topic name with max 6 words  based on the keywords: {', '.join(top_keywords)}"

        response = palm.generate_text(
            model="models/text-bison-001",  
            prompt=input_text,
            max_output_tokens=50
        )
        topic_name = response.result.strip()

        topic_keywords[topic_name] = top_keywords
    return topic_keywords


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyBSf9YAbtiT3PBEgDsuKJlr--pb9WZj1_w")
    
    # Load the FAISS index with dangerous deserialization allowed
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        google_api_key="AIzaSyBSf9YAbtiT3PBEgDsuKJlr--pb9WZj1_w"
    )

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def get_news(api_key1, keyword):
    try:
        url = f'https://newsapi.org/v2/everything?q={keyword}&apiKey={api_key1}'
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        articles = data.get('articles', [])  # Use .get() to safely access the 'articles' key
        news_list = []
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            link = article.get('url', '')
            news_list.append({'title': title, 'description': description, 'link': link})
        return news_list
    except Exception as e:
        print(f"An error occurred: {e}")
        return None





def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",google_api_key="AIzaSyBSf9YAbtiT3PBEgDsuKJlr--pb9WZj1_w")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])
    
def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3,google_api_key="AIzaSyBSf9YAbtiT3PBEgDsuKJlr--pb9WZj1_w")

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain
