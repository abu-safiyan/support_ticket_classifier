import streamlit as st
from predict import prediction


st.title('Category and Priority Prediction')
st.markdown('<h2><b>Support Ticket</b></h2>', unsafe_allow_html=True)
st.markdown('<h4>Subject</h4>', unsafe_allow_html=True)
subject = st.text_input('Subject', placeholder='Enter subject here...', label_visibility='collapsed')
st.markdown('<h4>Ticket Text</h4>', unsafe_allow_html=True)
ticket_text = st.text_area('Ticket Text', placeholder='Enter text here...', label_visibility='collapsed')

if st.button('Predict'):
    if not subject.strip() or not ticket_text.strip():
        st.error('Please fill in both Subject and Ticket text.')
    else:
        category, priority = prediction(subject, ticket_text)
        result_box = st.container()
        with result_box:
            st.write('Predicted Category:', category)
            st.write('Given Priority: ', priority)
