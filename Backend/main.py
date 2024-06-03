from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from flask import render_template 
from bot import ChatBot


app = Flask(__name__)
#CORS(app)

chat_resp = {}

@app.route("/q/<query>")
@cross_origin()
def get_resp(query):
    
    obj = ChatBot()
    text = obj.ask_bot(query)

    return jsonify(text),200

if __name__ == "__main__":
    app.run(debug=True)




