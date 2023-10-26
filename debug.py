import streamlit as st

button1 = st.button('Check 1')

if st.session_state.get('button') != True:

    st.session_state['button'] = button1

if st.session_state['button'] == True:

    st.write("button1 is True")

    if st.button('Check 2'):

        st.write("Hello, it's working")

        st.session_state['button'] = False

        st.checkbox('Reload')