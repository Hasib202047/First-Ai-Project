import traceback
from src.mcqgenerator.utils import read_file
import streamlit as st
from langchain_community.callbacks import get_openai_callback
from src.Summarization import summarization_chain

# Creating a title for the app
st.title('Text Summarization')

# Create a form using st.form
with st.form('user_inputs'):
    # File upload 
    uploaded_file=st.file_uploader('Upload a PDF or txt file')

    words = st.number_input('input the number of word in summarization', min_value= 50, max_value= 500)

    # Add button
    button=st.form_submit_button('Summarize')

    # Check if the button is clicked and all fields have input 
    if button and uploaded_file is not None and words:
        with st.spinner('loading...'):
            try:
                text=read_file(uploaded_file)

                # Count tokens and  the cost of API call
                with get_openai_callback() as cb:
                    response = summarization_chain.invoke({
                            'text': text,
                            'words': words
                        })
            
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error('Error')

            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")
                
                st.write(response)