import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="VIT Campus Assistant", layout="wide")
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

st.sidebar.title("Settings")
if st.sidebar.button("Clear History"):
    st.session_state.messages = []
    st.session_state.chat = None
    st.rerun()

try:
    f = open("campus_data.txt", "r")
    context = f.read()
except:
    context = ""

st.title("VIT Campus ASsistan")

if "messages" not in st.session_state:
    st.session_state.messages = []
    instruction = "You are a helpful assistant for VIT Vellore students, use this data: " + context
    st.session_state.chat = model.start_chat(history=[
        {"role": "user", "parts": [instruction]},
        {"role": "model", "parts": ["Okay, I am ready."]}
    ])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask me anything you want.")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    response = st.session_state.chat.send_message(user_input)
    
    with st.chat_message("assistant"):
        st.write(response.text)

    st.session_state.messages.append({"role": "assistant", "content": response.text})


