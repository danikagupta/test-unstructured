import streamlit as st

from utils import show_navigation
show_navigation()

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Jasminum_officinale_-_Bot._Mag._31%2C_1787.jpg/220px-Jasminum_officinale_-_Bot._Mag._31%2C_1787.jpg");
             background-attachment: fixed;
             background-size: cover;
             opacity: 0.9;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# add_bg_from_url() 

  
st.markdown("# HELLO WORLD")

st.markdown("""
             Remember to set the following in your .streamlit/secrets.toml file:
             * OPENAI_API_KEY
            * PINECONE_API_KEY, PINECONE_API_ENV, PINECONE_INDEX_NAME
            * If using EMAIL, SENDGRID_API_KEY
            * If using Dataframe, DATAFRAME_CSV as the URL for the corresponding CSV file 
            * If using the Quiz App, QUIZ_CSV as the name of the CSV file with Quiz data....
             """)

