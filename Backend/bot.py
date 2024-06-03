import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import numpy as np
import pandas as pd
import json
import string
import random
import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class ChatBot:

  def __init__(self):
    pass

  def text_preprocess(self, inp_text, translator, lemmatizer, stop_words):
    #STEP1: The letters of the sentence is coverted to a lower case
    i = inp_text.lower()
    #STEP2: Remove all the punctuations
    sentence = i.translate(translator) 
    #STEP3: Split the words in the sentence to process those further (vector)
    sentence = sentence.split()
    #STEP4: Remove all stopwords
    stripped_sentence = [y for y in sentence if y not in stop_words]
    #STEP5: Lemmatize each word in the sentence (stop words removed)
    output = []
    for j in stripped_sentence:
      word = lemmatizer.lemmatize(j)
      output.append(word)

    #DEBUG=TRUE
    #print("inp_text -> ", inp_text, " stripped_sentence -> ", stripped_sentence, " output -> ", output)

    return output


  def ask_bot(self, query):

    if len(query)<1:
      chat_resp = {
        "q": query,
        "c": "Error:<none>",
        "a": "I did not get any valid query, Please enter your query !",
        "t": "Error:<none>",
        "s": "Error:<none>",
      }

      return chat_resp

    #query = "What strategies can help adults with ADHD stay ADHD"
    #query = "Who are some famous figure2s with ADHD?"

    with open ('prod_intents.json') as content:
      intent_content = json.load(content)

    tags = []
    tagsforinputs = []
    responses = []
    single_resp = []
    input = []
    rulebase_i_t = []
    cleaned_query = ""

    #Extract Tags and Responses and create lists (one to one map "tag -> Response")
    for intent in intent_content['intents']:
      tags.append(intent['tag'])
      responses.append(intent['tag'])
      responses.append(intent['responses'])
      single_resp.append(intent['responses'])

    for intent in intent_content['intents']:
      for lines in intent['patterns']:
        input.append(lines)
        tagsforinputs.append(intent['tag'])


    #Store data in a table form using pandas DataFrame
    data = pd.DataFrame({"tags": tags, "responses": single_resp})
    data_i_t = pd.DataFrame({"inputs": input, "tags": tagsforinputs})


    le = LabelEncoder()
    y_train = le.fit_transform(data['tags'])

    saved_model = keras.models.load_model('chatbot_2.keras')

    #Text Preprocessing
    stop_words = set(stopwords.words('english'))
    translator = str.maketrans('', '', string.punctuation) 
    lemmatizer = WordNetLemmatizer()

    index_counter = 0
      
    for i in data_i_t['inputs']:
      processed_sentence = self.text_preprocess(i, translator, lemmatizer, stop_words)

      s = ' '.join([str(e) for e in processed_sentence])
      rulebase_i_t.append(s)
      rulebase_i_t.append(data_i_t.iat[index_counter,1])

      #Update the relevant input cell in the DataFrame
      data_i_t.at[index_counter, 'inputs'] = s
      index_counter = index_counter + 1

    #Text Preprocessing - Tokenization
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=2500)
    tokenizer.fit_on_texts(data_i_t['inputs'])
    train = tokenizer.texts_to_sequences(data_i_t['inputs'])
    
    processed_sentence = self.text_preprocess(query, translator, lemmatizer, stop_words)
    processed_sentence = ' '.join([str(e) for e in processed_sentence])
    cleaned_query = processed_sentence

    ##Rulebase - Check if the query matches directly with the intentes Patterns
    rulebase_resp = ''
    rulebasematch = 1
    try:
      rulebase_resp = rulebase_i_t.index(processed_sentence)
    except ValueError:
      rulebasematch = 0

    if rulebasematch==1:
      rulebase_resp_found_tag = rulebase_i_t[rulebase_resp + 1]
      selected_resp = responses.index(rulebase_resp_found_tag)
      answer = random.choice(responses[selected_resp+1])

      chat_resp = {
        "q": query,
        "c": cleaned_query,
        "a": answer,
        "t": rulebase_resp_found_tag,
        "s": "rule_base"
      }

      return chat_resp
    ##EndOf RuleBase

    processed_sentence = processed_sentence.split()

    train_query_input = tokenizer.texts_to_sequences(processed_sentence)
    train_query_input_2D = []
    train_query_input_2D.append(train_query_input)

    #Check for unknown words and then throw an error
    for i in train_query_input_2D[0]:
      if len(i) == 0:
        answer = "Sorry, I dont understand your query. Please try again with a different query."
        chat_resp = {
        "q": query,
        "c": cleaned_query,
        "a": answer,
        "t": 'error:unknown_word_included',
        "s": "error_handling"
        }

        return chat_resp
    
    #Trainer ANN has 8 inputs, therefore the number of inputs are hardcoded rather than recalculation it based on the intent file to save time
    train_query_input = tf.keras.preprocessing.sequence.pad_sequences(train_query_input_2D, 8)

    output = saved_model.predict(train_query_input)
    output = output.argmax()

    response_tag = le.inverse_transform([output])[0]
    tagindex = responses.index(response_tag)
    answer = random.choice(responses[tagindex+1])

    chat_resp = {
        "q": query,
        "c": cleaned_query,
        "a": answer,
        "t": response_tag,
        "s": 'ml_model'
    }

    return chat_resp
