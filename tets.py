import streamlit as st

st.title("Hello, Streamlit!")
st.write("chose your language")
lang=st.selectbox("coding language: ",["c","cpp","pyhton","java","shell"])
st.success(f" {lang} is a good choice")

st.sidebar.title("Sidebar")