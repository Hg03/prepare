__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import pandas as pd
from sklearn import model_selection, preprocessing, ensemble
from markdownlit import mdlit
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA, VectorDBQA
from langchain.memory import ConversationBufferMemory
import joblib
st.set_page_config(page_title='Placement',layout='wide',page_icon='boat')



def campus_recruitment():
    st.title('Campus Recruitment Prediction')

    model = joblib.load('models/campus_recruitment_prediction.joblib')

    data = {'gender':[],'ssc_p':[],'ssc_b':[],'hsc_p':[],'hsc_b':[],'hsc_s':[],'degree_p':[],'degree_t':[],'workex':[],'etest_p':[],'specialisation':[],'mba_p':[]}

    with st.form('myform'):
        
        gender = st.radio('Select your gender',['Male','Female'])
        if gender == 'Male':
            data['gender'].append('M')
        else:
            data['gender'].append('F')
        with st.container():
            c1,c2 = st.columns(2)
            with c1:
                ssc_p = st.number_input('Senior Secondary %',min_value=0,max_value=100)
                data['ssc_p'].append(ssc_p)
            with c2:
                ssc_b = st.selectbox('Senior Secondary Branch',['Central','Others'])
                data['ssc_b'].append(ssc_b)
        with st.container():
            c1,c2,c3 = st.columns(3)
            with c1:
                hsc_p = st.number_input('Higher Secondary %',min_value=0,max_value=100)
                data['hsc_p'].append(hsc_p)
            with c2:
                hsc_b = st.selectbox('Higher Secondary Branch',['Central','Others'])
                data['hsc_b'].append(hsc_b)
            with c3:
                hsc_s = st.selectbox('Higher Secondary Subject',['Science','Commerce','Arts'])
                data['hsc_s'].append(hsc_s)
        with st.container():
            c1,c2 = st.columns(2)
            with c1:
                degree_p = st.number_input('Degree %',min_value=0,max_value=100)
                data['degree_p'].append(degree_p)
            with c2:
                degree_t = st.selectbox('Degree Branch',['Comm&Mgmt', 'Sci&Tech', 'Others'])
                data['degree_t'].append(degree_t)
        workexp = st.checkbox('Do you have work experience',value=bool)
        workex = 'Yes' if workexp else 'No'
        data['workex'].append(workex)
        etest_p = st.number_input('etest p',min_value=0,max_value=100)
        data['etest_p'].append(etest_p)
        specialisation = st.selectbox('Specialisation',['Mkt&Fin', 'Mkt&HR'])
        data['specialisation'].append(specialisation)
        mba_p = st.number_input('MBA %',min_value=0,max_value=100)
        data['mba_p'].append(mba_p)
        
        if st.form_submit_button('Predict that Student is recruited or not ??'):
            feature = pd.DataFrame.from_dict(data)
            prediction = model.predict(feature)
            with st.expander('Data you entered'):
                st.dataframe(feature)
            if prediction[0] == 'Placed':
                mdlit('### Congratulations 🎉, you are [green]placed[/green]')
            else:
                mdlit("### Sorry, But it's just a beginning ,you are [red]not placed[/red]")

def placement_bot():
    st.title('Placement Bot')
    st.info('For now, TCS, Cognizant, Tech Mahindra, Cisco')
    persist_directory = 'chromadb'
    memory = ConversationBufferMemory()
    openai_api_key = st.sidebar.text_input('Enter your API key',type='password')
    if openai_api_key:
        embedding = OpenAIEmbeddings(openai_api_key = openai_api_key)
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        qa = VectorDBQA.from_chain_type(llm=OpenAI(openai_api_key = openai_api_key), chain_type="stuff", vectorstore=vectordb,memory=memory)
        
        
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"],avatar="👾"):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask your query",role="🌕"):
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.chat_message("user",role="👾"):
                st.markdown(prompt)
            with st.chat_message("assistant",role="🤖"):
                message_placeholder = st.empty()
                full_response = ""
                response = qa.run(prompt)
                message_placeholder.markdown(response)
            st.session_state.messages.append({"role":"assistant","content":response})  
    else:
        st.warning('Enter your OpenAI API Key through sidebar')  
    # query = st.text_area('Enter your query')
    # if st.button('Ask the Bot'):
    #     response = qa.run(query)
    #     st.code(response)

def main():
    two_opt = st.sidebar.selectbox('Select upon two projects',['Campus Recruitment Prediction','Placement Bot'])
    if two_opt == 'Campus Recruitment Prediction':
        campus_recruitment()
    elif two_opt == 'Placement Bot':
        placement_bot()
        
main()
                
        
                
        
