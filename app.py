from flask import Flask,request,flash
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import os
import json
from flask_session import Session
from single_image_object_counting import detect_image
app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/detect-image',methods = ['POST'])
def login():
        if 'file' not in request.files:
            flash('No file part')
            return 'ERROR'

        file = request.files['file']

        if file.filename == '':
            flash('No file selected for uploading')
            return 'ERROR'

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join('uploads', filename))
            response = detect_image(os.path.join('uploads', filename))
            # flash('File successfully uploaded')
            return {"data":response}
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return 'ERROR'
if __name__ == '__main__':
    app.config["SESSION_PERMANENT"] = False
    app.config["SESSION_TYPE"] = "filesystem"
    Session(app)
    app.run(debug = True)