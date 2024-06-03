import sys

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from flask import render_template 
from termcolor import colored
from bot import ChatBot

#To invoke the server please pass the argument server
#If you want to access the CLI with debug mode, pass argument debug

if len(sys.argv) == 2:
    command = sys.argv[1]
else:
    command = "none"

if command == 'server':

    print("")
    print(colored("Focus Friend ChatBot Server (1.0V) is being intialized.............", 'yellow', attrs=['reverse', 'blink','bold']))
    print("")

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

else:
    input_query = ""

    print("")
    welcome_text = colored('===== Hello Friend, I am Focus Friend ChatBot =====', 'green', attrs=['blink'])
    print(welcome_text)
    print("")
    welcome_text = colored('Note: type exit to close the chat', 'red')
    print(welcome_text)
    print("")

    while input_query != "exit":
        print(colored("Enter Your Query :","yellow"))
        input_query = input()
        if input_query == "exit":
            break
        else:
            obj = ChatBot()
            text = obj.ask_bot(input_query)
            print(colored("Focus Friend Answer :", "light_green"), text['a'])
            if command == "debug":
                print(colored("===============debug mode===============",'dark_grey'))
                print(colored("Query => ", 'dark_grey'), text['q'])
                print(colored("Cleaned => ", 'dark_grey'),text['c'])
                print(colored("Tag => ", 'dark_grey'), text['t'])
                print(colored("Source => ", 'dark_grey'), text['s'])
                print(colored("========================================",'dark_grey'))
        print("")

    print("")
    print(colored("Bye", 'yellow', attrs=['reverse', 'blink','bold']))







