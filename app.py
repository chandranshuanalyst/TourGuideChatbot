# Q&A Chatbot
from langchain_openai import OpenAI

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import streamlit as st
import os


## Function to load OpenAI model and get respones

def get_openai_response(question):
    llm=OpenAI(openai_api_key=os.environ["OPEN_API_KEY"],temperature=0.5)
    response=llm.invoke(question)
    return response

##initialize our streamlit app

st.set_page_config(page_title="Q&A Chatbot")

st.header("By - Chandranshu Jain")

input=st.text_input("Question: ",key="input")
response=get_openai_response(input)

submit=st.button("Answer")

## If ask button is clicked

if submit:
    st.subheader("The Answer is")
    st.write(response)
