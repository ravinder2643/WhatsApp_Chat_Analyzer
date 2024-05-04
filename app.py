#eske dono sentiment and visualization running
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import requests


import pandas as pd
import numpy as np
import time
import math
import matplotlib.dates as mdates



import google.generativeai as genai 
import seaborn as sns

from helper import get_news
from preprocessor import preprocess_text_file
from helper import stats, most_busy_user, get_wordcloud, top_com_words, fetch_message, top_emoji, time_line, daily_timeline, user_input, week_activity, month_activity, heat_map_data,monthly_senti_change,daily_senti_change,monthly_emotion_change,daily_emotion_change,compount_sentiment_monthly,compount_emotion_monthly,subjectivity_percentage,subjectivity_trend,chat_keywords




genai.configure(api_key="AIzaSyBSf9YAbtiT3PBEgDsuKJlr--pb9WZj1_w")
def display_topic_keywords(topic_keywords):
    columns = st.columns(5)

    for i, (topic_name, keywords) in enumerate(topic_keywords.items()):
        columns[i % 5].subheader(f"**{topic_name}**")
        
        columns[i % 5].markdown(", ".join(keywords[:5]))
proceed = False
st.set_page_config(page_title="Chat Analyzer", page_icon=":speech_balloon:", layout="wide")
uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
show=True
if uploaded_file is not None:
    show=False
    df = preprocess_text_file(uploaded_file)

    cf = df
    user_list = cf['user'].unique().tolist()
    # user_list.remove("group-notification")
    user_list.sort()
    user_list = ['Overall'] + user_list
    st.text(f"Number of users: {len(user_list)-1}")

    selected_user = st.selectbox("Show analysis with respect to", user_list)
    proceed=True
    
else:
    st.title("Upload Any Whatsapp conversation to get started")
    col1,col2= st.columns(2)
    with col1:
        st.header("Export the chat without media")
        url = "https://github.com/ravinder2643/WhatsApp_Chat_Analyzer"
        st.markdown(f"[Link to source code]({url})", unsafe_allow_html=True)
    with col2:    
        local_image_path = "./img.webp"
        width = 300
        st.image(local_image_path, caption="Export the chat without media",width=width)

    
    
   

while not proceed:
    time.sleep(1)
tabs = st.tabs(["Data Visualization", "Sentiment Analysis", "Recommendations on Interest", "ChatBot"])
with tabs[0]:
    st.header("Data Visualization")
    st.header((f"{selected_user} Analysis"))
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.header(("Total Messages"))
        st.title(stats(selected_user, df)[0])

    with col2:
        st.header(("Total Words"))
        st.title(stats(selected_user, df)[1])

    with col3:
        st.header(("Total Media Shared"))
        st.title(stats(selected_user, df)[2])

    with col4:
        st.header(("Total Links Shared"))
        st.title(stats(selected_user, df)[3])

    st.header(('Timeline Graph'))
    col_timeline1, col_timeline2 = st.columns(2)

    with col_timeline1:
        st.subheader(("Monthly Graph"))
        fig_monthly = px.line(time_line(selected_user, df), x='time', y='message', template='plotly_dark')
        st.plotly_chart(fig_monthly)

    with col_timeline2:
        st.subheader(("Daily Graph"))
        fig_daily = px.line(daily_timeline(selected_user, df), x='only_date', y='message', template='plotly_dark')
        st.plotly_chart(fig_daily)

    st.header(("Activity Map"))
    col_activity1, col_activity2 = st.columns(2)
    
    with col_activity1:
        st.subheader(("Most Busy Months"))
        month_data = month_activity(selected_user, df)
        month_data = month_data.sort_values(by="month")
        month_data["month"] = pd.Categorical(
            month_data["month"],
            categories=[
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ],
            ordered=True,
        )

        st.bar_chart(month_data, x="month", y="message", use_container_width=True)

    with col_activity2:
        st.subheader(("Most Busy Day"))
        week_data = week_activity(selected_user, df)
        week_data = week_data.sort_values(by="dayname")
        week_data["dayname"] = pd.Categorical(
            week_data["dayname"],
            categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            ordered=True,
        )

        st.bar_chart(week_data, x="dayname", y="message", use_container_width=True)

    if selected_user == 'Overall':
        st.header(("Most Busy Users"))
        most_busy_users, busy_user_df = most_busy_user(df)
        col_busy1, col_busy2 = st.columns(2)

        with col_busy2:
            st.bar_chart(most_busy_users, use_container_width=True)

        with col_busy1:
            st.dataframe(busy_user_df)

    else:
        st.header((f"{selected_user}'s messages in the group"))
        st.dataframe(fetch_message(df, selected_user))

    st.header(("User Activity Heatmap"))
    heat_map = heat_map_data(selected_user, df)
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(25, 10))

    day_names_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat_map = heat_map.reindex(day_names_order)

    sns.heatmap(heat_map, cbar_kws={'label': 'Message Count'}, annot=False, ax=ax_heatmap)

    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=90, ha='right')
    ax_heatmap.set_ylabel('Days')

    ax_heatmap.set_xlabel('Time Range')
    st.pyplot(fig_heatmap)


    st.header(("WordCloud"))
    df_wc = get_wordcloud(selected_user, df)
    fig_wc, ax_wc = plt.subplots()
    ax_wc.imshow(df_wc)
    st.pyplot(fig_wc)

    st.header(("Most Common Words"))
    col_common1, col_common2 = st.columns(2)
    top_words_df = top_com_words(selected_user, df)

    with col_common1:
        st.dataframe(top_words_df)

    with col_common2:
        st.bar_chart(top_words_df, x="words", y="frequency", use_container_width=True)

    # st.header(("Most Common Emojis"))
    # col_emoji1, col_emoji2 = st.columns(2)
    # top_emojis_df = top_emoji(selected_user, df)

    # if top_emojis_df.empty:
    #     st.header("Most Common Emojis")
    #     st.write("No Emoji analysis possible.")
    # else:
    #     st.header("Most Common Emojis")
    #     col_emoji1, col_emoji2 = st.columns(2)

    #     with col_emoji1:
    #         st.dataframe(top_emojis_df)

    #     with col_emoji2:
    #         st.bar_chart(top_emojis_df, x="emoji", y="frequency", use_container_width=True)
            
            
    # Assuming top_emojis_df is your DataFrame containing emoji frequencies

# Fetch and display the top emojis DataFrame
    top_emojis_df = top_emoji(selected_user, df)

    if top_emojis_df.empty:
        st.header("Most Common Emojis")
        st.write("No Emoji analysis possible.")
    else:
        st.header("Most Common Emojis")
        col_emoji1, col_emoji2 = st.columns(2)

        with col_emoji1:
            st.dataframe(top_emojis_df)

        with col_emoji2:
            # Create a pie chart
            fig, ax = plt.subplots()
            ax.pie(top_emojis_df['frequency'],  autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            # Display the pie chart using Streamlit
            st.pyplot(fig)
        
       

with tabs[1]:
    st.header("Sentiment Analysis")
    col_trend1, col_trend2= st.columns(2)
    with col_trend1:
        monthly_trend= monthly_senti_change(selected_user,df)
        monthly_trend.index = monthly_trend.index.astype(str)
        all_sentiments = ['Joy', 'Anger', 'Neutral', 'Sadness']

        all_sentiments_df = pd.DataFrame(columns=all_sentiments)
        all_sentiments_df.reset_index(inplace=True)
        monthly_trend.reset_index(inplace=True)

# Then perform the merge
#monthly_trend = pd.merge(all_sentiments_df, monthly_trend, how='outer', on='index')
        monthly_trend = pd.merge(all_sentiments_df.reset_index(), monthly_trend.reset_index(), how='outer', left_index=True, right_index=True)

        #monthly_trend = pd.merge(all_sentiments_df, monthly_trend, how='outer', left_index=True, right_index=True)

        monthly_trend = monthly_trend.fillna(0)


        st.subheader('Sentiment Trend Over Months:')
        
        fig = px.line(monthly_trend.reset_index(), x='date', y=['positive', 'neutral', 'negative'],
                    labels={'value': 'Sentiment Proportion', 'date': 'Month'},
                    line_shape='linear', render_mode='svg')

        fig.update_layout(xaxis=dict(tickangle=-45, tickmode='array', tickvals=monthly_trend.index, ticktext=monthly_trend.index))

        st.plotly_chart(fig)
    with col_trend2:
        st.subheader('Sentiment Trend Over Dates:')
        daily_trend= daily_senti_change(selected_user,df)
        all_sentiments = ['Joy', 'Anger', 'Neutral', 'Sadness']

        all_sentiments_df = pd.DataFrame(columns=all_sentiments)

        daily_trend = pd.merge(all_sentiments_df, daily_trend, how='outer', left_index=True, right_index=True)

        daily_trend = daily_trend.fillna(0)
        fig = px.line(daily_trend, x='only_date', y=['positive', 'neutral', 'negative'],
              labels={'value': 'Sentiment Proportion', 'only_date': 'Date'},
              line_shape='linear', render_mode='svg'  )

        fig.update_xaxes(
            dtick='M2',  
            tickformat='%b %d',  
            tickangle=-45  
        )

        
        st.plotly_chart(fig)
    coul_trend1, coul_trend2= st.columns(2)
    with coul_trend1:
        monthly_trend= monthly_emotion_change(selected_user,df)
        monthly_trend.index = monthly_trend.index.astype(str)

        
        monthly_trend.index = monthly_trend.index.astype(str)

        
        st.subheader('Emotion Trend Over Months:')
        
        
        fig = px.line(monthly_trend.reset_index(), x='only_date', y=['Joy', 'Anger', 'Neutral', 'Sadness'],
                    labels={'value': 'Emotion Proportion', 'only-date': 'Month'},
                    line_shape='linear', render_mode='svg')

        
        fig.update_layout(xaxis=dict(tickangle=-45, tickmode='array', tickvals=monthly_trend.index, ticktext=monthly_trend.index))

        
        st.plotly_chart(fig)
    with coul_trend2:
        st.subheader('Emotion Trend Over Dates:')
        daily_trend= daily_emotion_change(selected_user,df)
        fig = px.line(daily_trend, x='only_date', y=['Joy', 'Anger', 'Neutral', 'Sadness'],
              labels={'value': 'Emotion Proportion', 'only_date': 'Date'},
              line_shape='linear', render_mode='svg'  )

        
        fig.update_xaxes(
            dtick='M2',  
            tickformat='%b %d',  
            tickangle=-45  
        )

        
        st.plotly_chart(fig)
        
        
        
        
    st.header('Mean Compound Sentiment Score:')
    data = compount_sentiment_monthly(selected_user,df)
    fig = px.line(data, x='only_date', y='polarity', title='Sentiment Trend Over Time',
        labels={'month': 'Timestamp', 'polarity': 'Mean Compound Sentiment Score'})

    fig.update_layout(height=600, width=1830)

    st.plotly_chart(fig)
        
    st.header('Mean Compound Emotion Score:')
    data = compount_emotion_monthly(selected_user,df)
    fig = px.line(data, x='only_date', y='emotion_nltk', title='Emotion Trend Over Time',
        labels={'month': 'Timestamp', 'emotion': 'Mean Compound Emotion Score'})
    
    
    fig.update_layout(height=600, width=1830)

    st.plotly_chart(fig)
    
    
    
    
    
    st.title('Sentiment vs Emotion_nltk Correlation Plot')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='sentiment', y='emotion_nltk', data=df, color='blue', label='Emotion_nltk')
    sns.scatterplot(x='sentiment', y='sentiment', data=df, color='red', label='Sentiment')

    z = np.polyfit(df['sentiment'], df['emotion_nltk'], 1)
    p = np.poly1d(z)
    beta_1, beta_0 = z
    theta = math.degrees(math.atan(beta_1))
    ax.plot(df['sentiment'], p(df['sentiment']), color='black', label='Correlation Line')

    correlation_coefficient = df['sentiment'].corr(df['emotion_nltk'])

    

    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Emotion_nltk')
    ax.set_title('Correlation between Sentiment and Emotion_nltk')

    ax.legend()

    st.subheader(f"The Correlation Coefficient is {round(correlation_coefficient, 3)}")
    st.subheader(f"Beta 0 (Intercept): {round(beta_0, 3)}")
    st.subheader(f"Beta 1 (Slope): {round(beta_1, 3)}")
    st.subheader(f"Angle of the slope (theta): {round(theta, 3)} degrees")

    st.pyplot(fig)


        
        
    
    sub1, sub2 = st.columns(2)
    with sub1:
        
        st.subheader('Subjectivity Percentage')
        st.bar_chart(subjectivity_percentage(selected_user, df))    
    with sub2:
        data= subjectivity_trend(selected_user,df)
        


        st.subheader('Subjectivity Trend Over Months:')
        
        st.plotly_chart(px.line(data, x='only_date', y=data.columns[1:], labels={'value': 'Subjectivity'}))
    
    

    st.header("Topic Modelling")
    st.header("User's interests")
    topic_keywords= chat_keywords(selected_user,df)
    display_topic_keywords(topic_keywords)
    

            

with tabs[2]:
   
    st.header("Recommadations of Events based on interest")
    keyword = st.text_input("Give any interest and get the latest related information")
    api_key1 = "2cd44d590df74759b11e706e7c3ab2ba"

    news = get_news(api_key1, keyword)
    
    # Display news articles title description and links
    if news:
        st.header(f"News related to '{keyword}':")
        for index, article in enumerate(news[:10], start=1):
            st.subheader(f"Article {index}:")
            st.write(f"Title: {article['title']}")
            st.write(f"Description: {article['description']}")
            st.write(f"Link: {article['link']}")
            st.write("*" * 150)
    else:
        st.write(f"No news found related to '{keyword}'.")


with tabs[3]:
    st.header("Chatbot using Gemini-Pro 💁")

    user_question = st.text_input("Ask any Question regading the app or analysis")

    if user_question:
        user_input(user_question)