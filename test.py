import streamlit as st


import google.generativeai as genai 

from dotenv import load_dotenv
genai.configure(api_key=os.getenv("GOOGLE_API"))
load_dotenv()
st.header("Chat with AI")
model=genai.GenerativeModel("gemini-pro")
chat=model.start_chat(history=[])
def get_response(que):
    res=chat.send_message(que,stream=True)
    return res
if 'chat_history' not in st.session_state:
    st.session_state['chat_history']=[]
input = st.text_input("Input",key="input")
submit = st.button("Ask The question")

if submit and input:
    res = get_response(input)
    st.session_state['chat_history'].append(("You",input))
    st.subheader("The Resposnse is ")
    for chunk in res:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("Bot",chunk.text))
st.subheader("The Chat History is ")

for role,text in st.session_state['chat_history']:
    st.write(f"{role}:{text}")