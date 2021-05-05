from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
from keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Pranshu.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
db = SQLAlchemy(app)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(80),  nullable=False)
    gen_lyrics= db.Column(db.String(1000), nullable=False)

    def __repr__(self):
        return f"{self.gen_lyrics}"

tokenizer = Tokenizer()
tokenizer=Tokenizer()
def preprocessing(data):
    corpus = data.lower().split('\n')

    tokenizer.fit_on_texts(corpus)
    
    total_words = len(tokenizer.word_index)+1
    input_sequences=[]
    for line in corpus:
        # line = re.sub(r'[^\w\s]', '', line)
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
    ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
    return (xs, ys, max_sequence_len, total_words)
def model(xs, ys, max_sequence_len, total_words, epochs):
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(150)))
    model.add(Dense(total_words, activation='softmax'))
    adam = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy',
                        optimizer=adam, metrics=['accuracy'])
    history = model.fit(xs, ys, epochs=12, verbose=1)
    # word = generate_text, seed_text, next_word, max_seq_len, model)
    return model


def generate_text(seed_text, max_seq_len, next_words, model):
    for i in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    print(seed_text)
    return seed_text
@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        
        try:
            data=open('data1.txt').read()
            X,Y,max_len,total_w=preprocessing(data)
            
            working_model=model(X,Y,max_len,total_w,100)
            content=request.form['content']
            output= generate_text(content,max_len,30,working_model)
            music=User(content=content,gen_lyrics=output)
            db.session.add(music)
            db.session.commit()
            
            
        except:
            return 'There was an error in adding your task'
        lyrics=User.query.filter_by(id=User.id).first()
        return render_template('results.html',lyrics=lyrics)
    else:
        return render_template('index.html')
    
      
       


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
  