import tensorflow as tf
import numpy as np
import pandas as pd
import json
import string

from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def text_preprocess(inp_text, translator, lemmatizer, stop_words):
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

def train_ann():

    EPOCHS = 1000

    #Text Preprocessing

    with open ('prod_intents.json') as content:
      intent_content = json.load(content)

    tags = []
    inputs = []

    #Extract Tags and Patterns and create lists (one to one map "input -> tag")
    for intent in intent_content['intents']:
      for lines in intent['patterns']:
        inputs.append(lines)
        tags.append(intent['tag'])

    #Store data in a table form using pandas
    data = pd.DataFrame({"inputs": inputs, "tags": tags})

    #Text Preprocessing
    stop_words = set(stopwords.words('english'))
    translator = str.maketrans('', '', string.punctuation) 
    lemmatizer = WordNetLemmatizer()
    
    index_counter = 0
    
    for i in data['inputs']:
      processed_sentence = text_preprocess(i, translator, lemmatizer, stop_words)

      #Update the relevant input cell in the DataFrame
      processed_sentence = ' '.join([str(e) for e in processed_sentence])
      data.at[index_counter, 'inputs'] = processed_sentence
      index_counter = index_counter + 1

    #Text Preprocessing - Tokenization
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=2500)
    tokenizer.fit_on_texts(data['inputs'])
    train = tokenizer.texts_to_sequences(data['inputs'])
    
    #Text Preprocessing - padding
    x_train = tf.keras.preprocessing.sequence.pad_sequences(train)

    #encoding tags (output of the ANN)
    le = LabelEncoder()
    y_train = le.fit_transform(data['tags'])

    print(y_train)
    return 0

    input_length = x_train.shape[1] #2D array - columns
    print('x_train.shape', x_train.shape)

    vocab = len(tokenizer.word_index)
    output_length = le.classes_.shape[0]
    print('le.classes_.shape', le.classes_.shape)

    no_of_tags = len(set(data['tags']))

    ## Construst ANN model##
    i = tf.keras.layers.Input(shape=(input_length,), name = "chatbot_input_layer") #input layer
    x = tf.keras.layers.Embedding(vocab+1,10)(i)
    x = tf.keras.layers.LSTM(no_of_tags, return_sequences=True)(x)
    x = tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Dense(100, activation="relu")(x)
    x = tf.keras.layers.Dense(output_length, activation="softmax")(x)

    model = tf.keras.Model(inputs=i, outputs=x)
    model.compile(loss = "sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    train = model.fit(x_train, y_train, epochs = EPOCHS)

    #save the trained model
    model.save('chatbot_2.keras')



    

train_ann()