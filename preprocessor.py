
import pandas as pd
from datetime import datetime
import numpy as np
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
def get_sentiment_polarity(message):
    analysis = TextBlob(message)
    return analysis.sentiment.polarity
def get_time_period(hour):
    if 0<=hour<1:
        return "12AM - 1AM"
    elif 1<= hour <11:
        return f'{str(hour)}AM - {str(hour+1)}AM'
    elif hour == 11:
        return '11AM - 12PM'
    elif hour == 12:
        return '12PM - 1PM'
    elif hour == 23:
        return f'{str(hour-12)}PM - 12AM'
    elif 12 < hour <23:
        return f'{str(hour-12)}PM - {str(hour-11)}PM'
def get_time_period_dotted(hour):
    if 0<=hour<1:
        return "12AM - 1AM"
    elif 1<= hour <11:
        return f'{str(hour)[:-2]}AM - {str(hour+1)[:-2]}AM'
    elif hour == 11:
        return '11AM - 12PM'
    elif hour == 12:
        return '12PM - 1PM'
    elif hour == 23:
        return f'{str(hour-12)[:-2]}PM - 12AM'
    elif 12 < hour <23:
        return f'{str(hour-12)[:-2]}PM - {str(hour-11)[:-2]}PM'
def extract_info(line):
    date_formats = ['%d/%m/%y, %I:%M %p', '%d/%m/%y, %H:%M -']
    
    for date_format in date_formats:
        try:
            date_obj = datetime.strptime(line[:17], date_format)
            message = line[17:].strip()
            return date_obj, message
        except ValueError:
            continue

    return None

# Function to preprocess the text file
def preprocess_text_file(uploaded_file):
    content = uploaded_file.read().decode("utf-8").splitlines()

    processed_lines = []
    current_date = None
    current_message = ""

    for line in content:
        line = line.strip()
        if not line:
            continue

        extracted_info = extract_info(line)
        if extracted_info:
            if current_message:
                processed_lines.append((current_date, current_message))
                current_message = ""
            current_date, current_message = extracted_info
        else:
            current_message += ' ' + line

    if current_message:
        processed_lines.append((current_date, current_message))

    df = pd.DataFrame(processed_lines, columns=['date', 'usermessage'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date'] = df['date'].dt.strftime('%d/%m/%y %I:%M %p')
    df['usermessage'] = df['usermessage'].str.lstrip('-')
    df[['user', 'message']] = df['usermessage'].str.extract(r'([^:]+):?(.*)')
    df.drop('usermessage', axis=1, inplace=True)
    df.loc[df['message'] == '', 'message'] = df['user']
    df.loc[df['message'] == df['user'], 'user'] = 'group-notification'
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y %I:%M %p')

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.strftime('%B')
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['am/pm'] = df['date'].dt.strftime('%p')
    df['dayname']=df['date'].dt.day_name()
    df['period'] = df['hour'].apply(get_time_period)
    change=False
    for i in df['period'].unique():
        if '.' in str(i):
            change=True
            break
    if change:
        df.drop('period',axis=1)
        df['period'] = df['hour'].apply(get_time_period_dotted)

    df['only_date']=df['date'].dt.date
    df = df[~df['user'].str.contains('added')]
    df = df[~df['user'].str.contains('left')]
    df = df[~df['user'].str.contains('security')]
    df = df[~df['user'].str.contains('changed')]
    df = df[~df['user'].str.contains('removed')]
    df = df[~df['user'].str.contains('deleted')]
    df = df[~df['user'].str.contains('joined')]
    df = df[~df['user'].str.contains('created')]

    df = df[~np.isnan(df['year'])]
    df['year'] = df['year'].astype(int)
    
    # Sentiment Analysis
    df['sentiment'] = df['message'].apply(get_sentiment_polarity)

    df['sentiment_category'] = pd.cut(df['sentiment'], bins=[-1, -0.1, 0.1, 0.5, 1], 
    labels=['negative', 'neutral', 'mixed', 'positive'])
    
    
    df['emotion_nltk'] = df['message'].apply(get_emotion_nltk)
    df['emotion_label_nltk'] = df['emotion_nltk'].apply(map_to_emotion_label)

    
    
    df['polarity'] = df['message'].apply(get_polarity)
    
    
    df['subjectivity'] = df['message'].apply(classify_subjectivity)
    


    return df

def get_emotion_nltk(text):
    vader_lexicon_path = './vader_lexicon.txt'
    sia = SentimentIntensityAnalyzer(lexicon_file=vader_lexicon_path)
    sentiment_score = sia.polarity_scores(text)
    emotion = sentiment_score['compound']
    return emotion

def map_to_emotion_label(emotion_score):
    if emotion_score > 0.2:
        return 'Joy'
    elif emotion_score < -0.2:
        return 'Sadness'
    elif emotion_score < 0:
        return 'Anger'
    else:
        return 'Neutral'
    
def get_polarity(message):
    analysis = TextBlob(message)
    return analysis.sentiment.polarity


def classify_subjectivity(message):
    analysis = TextBlob(message)
    if analysis.sentiment.subjectivity > 0.5:  
        return 'Subjective'
    else:
        return 'Objective'