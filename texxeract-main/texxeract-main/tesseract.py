
import pandas as pd
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import boto3
import io
import pyttsx3
import base64
import spacy
import logging
import time
import cv2
import nltk
import re
import heapq
import pytesseract
from botocore.exceptions import ClientError
nlp = spacy.load('en_core_web_sm') 
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False,sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_csv(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download csv file</a>' # decode b'abc' => abc



@st.cache
def get_summary(detected_text):
                    # Removing Square Brackets and Extra Spaces
                    article_text = re.sub(r'\[[0-9]*\]', ' ', detected_text)
                    article_text = re.sub(r'\s+', ' ', article_text)
                    article_text = article_text.split(". ")
                    if len(article_text)<4:
                        return 0
                    else:
                        sumnum = len(article_text)//4
                        i=0
                        for at in article_text:
                            # Removing special characters and digits
                            bt = re.sub('[^a-zA-Z]', ' ', at )
                            article_text[i] = re.sub(r'\s+', ' ', bt)
                            i+=1
                        
                        stopwords = nltk.corpus.stopwords.words('english')

                        word_frequencies = {}
                        for ft in article_text:
                            for word in nltk.word_tokenize(ft):
                                if word not in stopwords:
                                    if word not in word_frequencies.keys():
                                        word_frequencies[word] = 1
                                    else:
                                        word_frequencies[word] += 1
                        maximum_frequncy = max(word_frequencies.values())

                        for word in word_frequencies.keys():
                            word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
                            
                        sentence_scores = {}
                        for sent in article_text:
                            for word in nltk.word_tokenize(sent.lower()):
                                if word in word_frequencies.keys():
                                    if len(sent.split(' ')) < 30:
                                        if sent not in sentence_scores.keys():
                                            sentence_scores[sent] = word_frequencies[word]
                                        else:
                                            sentence_scores[sent] += word_frequencies[word] 
                                            
                        summary_sentences = heapq.nlargest(sumnum, sentence_scores, key=sentence_scores.get)

                        summary = ' '.join(summary_sentences)
                        return summary



                  
def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
        object_to_download.encode("UTF-8")
        

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode("UTF-8")).decode()
    

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'





def detect_text(file_name,bucket):

    client=boto3.client('rekognition', region_name='ap-south-1',
                       aws_access_key_id='AKIAV6PGSYO67R2HTOF',
                       aws_secret_access_key='tFqLkJj1OxtcvVO8gQ8N5JLbWd6K0X8eiam+6JU')

    img = Image.open(file_name)

    # Create a buffer to hold the bytes
    buf = io.BytesIO()

    # Save the image as jpeg to the buffer
    img.save(buf, 'jpeg')

    # Rewind the buffer's file pointer
    buf.seek(0)

    # Read the bytes from the buffer
    image_bytes = buf.read()
    response = client.detect_text(Image={'Bytes':image_bytes})
        

    
    
    
    textDetections=response['TextDetections']
    #print ('Detected text\n----------')
    tokens = []
    for text in textDetections:
        #print(text['DetectedText'])
        if text['Type'] == 'LINE':
            tokens.append(text['DetectedText'])
    detected_text= " ".join((tokens))

    nlp = spacy.load('en_core_web_sm') 
    sentence = detected_text
    doc = nlp(sentence) 
    #for ent in doc.ents: 
      #  print('\n',ent.text, ent.start_char, ent.end_char, ent.label_) 
    return detected_text,doc
    
    






@st.cache
def upload_file(file_name, bucket, object_name):
    
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    
    # If S3 object_name was not specified, use file_name
    #if object_name is None:
     #   object_name = file_name
    
    
    c = Image.open(file_name)
    in_mem_file = io.BytesIO()
    c.save(in_mem_file, "JPEG")
    in_mem_file.seek(0)
    
    
    s3_client = boto3.client('s3', region_name='ap-south-1',
                       aws_access_key_id='AKIAV6PGSYO67CR2HOF',
                       aws_secret_access_key='tFqLkJj1xOxtvVO8gQ8N5JLbWd6K0X8eiam+6JU')
    try:
         response = s3_client.upload_fileobj(in_mem_file, bucket, object_name)   
    except ClientError as e:
        logging.error(e)
        return False
    return True

@st.cache
def get_keys(bucket):
    """
    will get the keys from aws s3
    Returns name of files
    
    """
    s3 = boto3.resource('s3', region_name='ap-south-1',
                       aws_access_key_id='AKIAV6PGSYO67CR2HOF',
                       aws_secret_access_key='tFqLkJj1xOxcvVO8gQ8N5JLbWd6K0X8eiam+6JU')
    bucket = s3.Bucket(bucket)
    
    contents = [_.key for _ in bucket.objects.all() ] 
    return contents


def read_images_from_s3(bucket, key):
    """Load image file from s3.

    Parameters
    ----------
    bucket: string
        Bucket name
    key : string
        Path in s3

    Returns
    -------
    np array
        Image array
    """
    s3 = boto3.resource('s3', region_name='ap-south-1',
                       aws_access_key_id='AKIAV6PGSYO67CR2HTF',
                       aws_secret_access_key='tFqLkJj1xxtcVO8gQ8N5JLbWd6K0X8eiam+6JU')
    bucket = s3.Bucket(bucket)
    object = bucket.Object(key)
    response = object.get()
    file_stream = response['Body']
    im = Image.open(file_stream)
    #imgplot = plt.imshow(im)
    #plt.show()
    st.image(im, caption=None, width=None, use_column_width=True, clamp=False, channels='RGB', output_format='auto', )
 













#background image for the app
#page_bg_img = '''
#<style>
#body {
#background-image: url("https://www.setaswall.com/wp-content/uploads/2018/04/Gifts-Snowflakes-Ribbon-Bowknot-1440x2880.jpg");
#background-size: cover;
#}
#</style>
#'''
#st.markdown(page_bg_img, unsafe_allow_html=True)


#left top details of app
st.sidebar.markdown('**ABOUT PROJECT :-**')
buffer1 = st.sidebar.checkbox('Made with\n')
if buffer1 ==1:
    st.sidebar.markdown('**Python** >3.6 version')
    st.sidebar.markdown('**Spacy** Name, Entity Recognition')
    st.sidebar.markdown('**Streamlit** This Magic ingredient here handles the Web part')
    st.sidebar.markdown('**AWS ec2** Deployment on the Cloud')
    st.sidebar.markdown("**AWS s3** Storage Bucket for USER'S data")
    st.sidebar.markdown('**AWS rekognition** Deep learning API')
   

buffer3 = st.sidebar.checkbox('See DEMO video')
if buffer3 ==1:
    st.sidebar.video('https://www.youtube.com/watch?v=4DVeGugV-5g&feature=youtu.be',format='mp4')                          


#title of our app
st.title('Texxeract :diamond_shape_with_a_dot_inside:')
st.subheader('An Interactive Web app to perform Text summarization and text detection on Handwritten articls.')



#secret keys we need to access the API  
key_token = 'AKIAV6PGSYO67CR2HTOF'
key_secret = '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'
     

def main():

    bucket = 'shivam3265bucket'
    option = st.selectbox( 'How would you like to be continue?',
                        ('Choose one','Try it out !!!!', 'See what others have uploaded'))
    if option is 'Try it out !!!!':
        user_name = st.text_input("Enter your Name here ")
        file = st.file_uploader(label = 'Upload an Image(.jpg) of article', type= ['PNG','JPEG','JPG'], accept_multiple_files=False, key='file_uploader')
        imgContain = st.radio(  "What does this image contain?",
                           ('Embedded text', 'Handwritten text'))
        
        if len(user_name)<3 or len(user_name) > 10:
            st.error('Enter a name of length >3 and <10')
        else:
            if  file :
                file_name= "2ndmajor" + "/"+ user_name
                msgbuf1 = "**" + file_name +"**" 
                st.markdown(msgbuf1)   
                if upload_file(file,'shivam3265bucket',file_name):
                    st.write("Uploading file to AWS S3 ")
                    my_bar = st.progress(0)
                    for percent_complete in range(100):
                        time.sleep(0.001)
                        my_bar.progress(percent_complete + 1)
                    st.image(file, caption=None, width=None, use_column_width=True, clamp=False, channels='RGB', output_format='auto', )
                if imgContain is 'Handwritten text':
                    detected_text, doc= detect_text(file,'shivam3265bucket')
                    st.markdown("**Here's the results **:point_down::")
                    attributes = {}
                    labels  = {}
                    i=0
                    for ent in doc.ents: 
                        attributes[i] = ent.text
                        labels[i] = ent.label_
                        i += 1
                    option1 = st.selectbox( 'Explore here',
                            (['Detected text','Important information']))
                    if option1 is 'Detected text':
                        st.write(detected_text)
                        if st.button('Play/Download the results'):
                            tmp_download_link = download_link(detected_text, 'detected text.txt', 'Click here to download .txt file')
                            
                            engine = pyttsx3.init()
                            engine.setProperty('rate', 190)
                            engine.save_to_file(detected_text, 'detectxt.mp3')
                            engine.runAndWait()
                            audio_file = open('detectxt.mp3', 'rb')
                            audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format='audio/ogg')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)
                            engine.stop()
                    elif option1 is  'Important information':
                        dff  = pd.DataFrame({
                            'Attribute': attributes,
                            'Label': labels
                            })
                        st.write(dff)    
                            
                        
                if imgContain is 'Embedded text':
                    dnew = pytesseract.image_to_string(Image.open(file))
                    st.markdown("**Here's the results **:point_down::")
                    option2 = st.selectbox( 'Explore here',
                            ('Detected text','Summary','Important information'))
                    if option2  is 'Detected text':
                        st.write(dnew)
                        if st.button('Play/Download the results'):
                            tmp_download_link = download_link(dnew, 'detected text.txt', 'Click here to download .txt file')
                                
                            engine = pyttsx3.init()
                            engine.setProperty('rate', 190)
                            engine.save_to_file(dnew, 'dnew.mp3')
                            engine.runAndWait()
                            audio_file = open('dnew.mp3', 'rb')
                            audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format='audio/ogg')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)
                            engine.stop()
                    elif option2 is 'Summary':
                        summary = get_summary(dnew)
                        if summary is 0:
                            st.write("Oops!!!! Too short to make a summary")
                        else:
                            st.write(summary)
                            if st.button('Play/Download the results'):
                                tmp_download_link = download_link(summary, 'detected summary.txt', 'Click here to download .txt file(SUMMARY)')
                            
                                engine = pyttsx3.init()
                                engine.setProperty('rate', 190)
                                engine.save_to_file(summary, 'summary.mp3')
                                engine.runAndWait()
                                audio_file = open('summary.mp3', 'rb')
                                audio_bytes = audio_file.read()
                                st.audio(audio_bytes, format='audio/ogg')
                                st.markdown(tmp_download_link, unsafe_allow_html=True)
                                engine.stop()
                    elif option2 is 'Important information':
                        nlp = spacy.load('en_core_web_sm') 
                        doc = nlp(dnew)
                        i =0
                        attributes1 = {}
                        labels1 = {}
                        for ent in doc.ents: 
                            attributes1[i] = ent.text
                            labels1[i] = ent.label_
                            i += 1

                        dff1  = pd.DataFrame({
                            'Attribute': attributes1,
                            'Label': labels1
                            })
                        st.write(dff1)
                        st.markdown(get_table_csv(dff1), unsafe_allow_html=True)
                        
                        
                                                

                    


            

                        
    elif option is   'See what others have uploaded':
        contents = get_keys('shivam3265bucket')
        i=1
        for content in contents:
            j = "**"
            j1 = j + str(i)
            i= i+1
            msg = j+content+"**"
            st.markdown(msg)
            read_images_from_s3("shivam3265bucket",content)
                    
                    
                 
    
    #text_count=detect_text(file,bucket)
    #print("Text detected: " + str(text_count))

    

    
    
if __name__ == "__main__":
    main()





st.write(' ')
st.write(' ')
st.write(' ')








st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
st.markdown('Made with :heart: by [Shivam](https://www.linkedin.com/in/shivam-purbia-b35a65166/?originalSubdomain=in)')
      
         

    
