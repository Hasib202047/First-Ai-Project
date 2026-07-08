import traceback
import streamlit as st
from langchain_community.callbacks import get_openai_callback
from src.SimpleApp import sequential_chain

# Creating a title for the app
st.title('Company Name suggestion with slogan!!!')

# Create a form using st.form
with st.form('user_inputs'):
    # Subject 
    subject=st.text_input('Insert Product Name',max_chars=40)

    # Add button
    button=st.form_submit_button('Generate Response')

    # Check if the button is clicked and all fields have input 
    if button and subject:
        with st.spinner('loading...'):
            try:
            
                # Count tokens and  the cost of API call
                with get_openai_callback() as cb:
                    result = sequential_chain.invoke({"product": subject})
            
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error('Error')

            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")
                st.write(result)
                    