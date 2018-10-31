import pyrebase
import platform
import os
import glob
from espeak import espeak
config={
    "apiKey": "apiKey",
    "authDomain": "fir-try-50bcd.firebaseio.com",
    "databaseURL": "https://fir-try-50bcd.firebaseio.com/",
    "storageBucket": "fir-try-50bcd.appspot.com"
}
def getName():
    Name=raw_input("enter name used in app")
    return Name
def stream_handler(message):
    if(message["data"] is not None):
        print(message["path"])
        print(message["data"])
        os.system("play -n synth 0.1 sin 880")
        #espeak.synth("beep")
        db.child("Alerts").child("haseel").remove()

firebase=pyrebase.initialize_app(config)
db=firebase.database()
my_stream = db.child("Alerts").child("haseel").stream(stream_handler)
while(1):
    continue
