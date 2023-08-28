from flask import Flask, Response, render_template,request,flash
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit
from werkzeug.datastructures import  FileStorage
import os
import cv2
import json
import base64
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from gender_age.detect import run
from utils import backbone
from flask_session import Session
from single_image_object_counting import detect_image
from webcam_counting import detect_video
from utils import visualization_utils as vis_util
app = Flask(__name__)
# socketio = SocketIO(app)
ALLOWED_EXTENSIONS_IMAGE = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
ALLOWED_EXTENSIONS_VIDEO = set(['mp4'])


def allowed_file_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_IMAGE
def allowed_file_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_VIDEO
@app.route('/detect-image',methods = ['POST'])
def fun_detect_image():
        if 'file' not in request.files:
            flash('No file part')
            return 'ERROR'

        file = request.files['file']

        if file.filename == '':
            flash('No file selected for uploading')
            return 'ERROR'

        if file and allowed_file_image(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join('uploads', filename))
            response = detect_image(os.path.join('uploads', filename))
            # flash('File successfully uploaded')
            return {"data":response}
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return 'ERROR'
@app.route('/detect-video',methods = ['POST'])
def fun_detect_video():
        if 'file' not in request.files:
            flash('No file part')
            return 'ERROR'

        file = request.files['file']

        if file.filename == '':
            flash('No file selected for uploading')
            return 'ERROR'

        if file and allowed_file_video(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join('uploads', filename))
            response = detect_video()
            data = {"age":{},"person":0,"gender":{}}
            # return response
            flash('File successfully uploaded')
            return 'OK'
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return 'ERROR'

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    try:
        data = request.json
        img_data = data.get('image')  # Sử dụng img_data thay vì image_data
        if img_data:
            # # Giải mã dữ liệu hình ảnh
            img_bytes = base64.b64decode(img_data.split(',')[1])
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            data = detect_image(frame=img)
            print(data)
            return json.dumps(data)
        else:
                return 'No image data received.'
    except Exception as e:
        print(f"An error occurred: {e}")
        return 'Internal Server Error', 500

def generate_frames():
    cap = cv2.VideoCapture(0)
    detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt')

    is_color_recognition_enabled = False # set it to true for enabling the color prediction for the detected objects
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            input_frame = frame
            image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
            font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.        
            counter, csv_line, counting_result = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                      input_frame,
                                                                                                      is_color_recognition_enabled,
                                                                                                      np.squeeze(boxes),
                                                                                                      np.squeeze(classes).astype(np.int32),
                                                                                                      np.squeeze(scores),
                                                                                                      category_index,
                                                                                                      use_normalized_coordinates=True,
                                                                                                   line_thickness=4)
            if(len(counting_result) == 0 and input == None):
                    cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
            else:
                data = run(input_frame)
                if(counting_result.find("'person:':") > 0):
                    data['person'] = counting_result.replace("'person:': ","")  
                else:
                     data['person'] = 0
                text = str(data['gender']).replace("{","").replace("}","") +' , '+ str(data['age']).replace("{","").replace("}","")
                result = counting_result +' , ' + text
                cv2.putText(input_frame,result, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
                        
            cv2.imshow('object counting',input_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    # return result
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_encoded = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_encoded + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.config["SESSION_PERMANENT"] = False
    app.config["SESSION_TYPE"] = "filesystem"
    # Session(app)
    # socketio.run(app)
    app.run(host='0.0.0.0', port=5000)
