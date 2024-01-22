# Q&A Chatbot
from langchain_openai import OpenAI

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import streamlit as st
import os

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain


## Function to load OpenAI model and get respones

def get_openai_response(country_name):
    llm=OpenAI(openai_api_key=os.environ["OPEN_API_KEY"],temperature=0.5)
    capital_template=PromptTemplate(input_variables=['country'],
    template="Please tell me the capital of the {country}")
    capital_chain=LLMChain(llm=llm,prompt=capital_template,output_key="capital")

    famous_template=PromptTemplate(input_variables=['capital'],
    template="Suggest me some amazing places to visit in {capital}")
    famous_chain=LLMChain(llm=llm,prompt=famous_template,output_key="places")

    eats_template=PromptTemplate(input_variables=['capital'],
    template="Suggest the top 5 famous dishes to eat in {capital}")
    eats_chain=LLMChain(llm=llm,prompt=eats_template,output_key="dishes")
    
    chain=SequentialChain(chains=[capital_chain,famous_chain,eats_chain],
    input_variables=['country'],
    output_variables=['capital',"places","dishes"])

    response=chain.invoke({'country':country_name})
    return response

##initialize our streamlit app

st.set_page_config(page_title="Q&A Chatbot")

st.header("January Capital Guide")

input=st.text_input("Enter Country Name: ",key="input")
response=get_openai_response(input)

submit=st.button("Answer")

## If ask button is clicked

if submit:
    st.subheader("The Answer is")
    st.write(response)
