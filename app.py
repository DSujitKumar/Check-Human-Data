from flask import Flask, send_from_directory, request,render_template
from flask import jsonify
import logging, os
from werkzeug import secure_filename
from os import walk
import tensorflow as tf
import numpy as np
import facenet
from align import detect_face
import cv2
import random
import string
import pickle
import configPath
# Configuring the path of pickle file 
pathPickle=configPath.pathPickle 
# Configuring the path of image Database file 
imageDB=configPath.imageDB
# Configuring the path of future image  file 
imageFuture=configPath.imageFuture

app = Flask(__name__)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#$$$$$$$$$$$$my code start  $$$$$$$$$$$$$$$$$$$$
# some constants kept as default from facenet
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44
input_image_size = 160
allImages={}
sess = tf.Session()

# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align')

# read 20170512-110547 model file downloaded from https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk
facenet.load_model("20170512-110547/20170512-110547.pb")

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

def getFace(img):
    faces = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.50:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                resized = cv2.resize(cropped, (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
                prewhitened = facenet.prewhiten(resized)
                faces.append({'face':resized,'rect':[bb[0],bb[1],bb[2],bb[3]],'embedding':getEmbedding(prewhitened)})
    return faces
def getEmbedding(resized):
    reshaped = resized.reshape(-1,input_image_size,input_image_size,3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding


def compare2face(face1,face2):
    if face1 and face2:
        # calculate Euclidean distance
        dist = np.sqrt(np.sum(np.square(np.subtract(face1[0]['embedding'], face2[0]['embedding']))))
        return dist
    
    return 10
#$$$$$$$$$$$$my code ends  $$$$$$$$$$$$$$$$$$$$

def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath



def id_generator(size=12, chars=string.digits):
	return ''.join(random.choice(chars) for _ in range(size))

def addData(imgId,ImgData,pathImg):
	global allImages
	if os.path.getsize(pathPickle) > 0:
		pickle_in=open(pathPickle,"rb")
		allImages=pickle.load(pickle_in)
	allImages.update({imgId:[ImgData,pathImg]})
	pickel_out=open(pathPickle,"wb")
	pickle.dump(allImages,pickel_out)
	pickel_out.close()

def delData(imgId):
	global allImages
	if os.path.getsize(pathPickle) > 0:
		pickle_in=open(pathPickle,"rb")
		allImages=pickle.load(pickle_in)
	Fpath=allImages[imgId][1]
	allImages.pop(imgId)
	pickel_out=open(pathPickle,"wb")
	pickle.dump(allImages,pickel_out)
	pickel_out.close()
	return Fpath
@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')
@app.route('/check', methods = ['POST'])
def api_root():
	global allImages #code to use global variable
	app.logger.info(PROJECT_HOME)
	if request.method == 'POST' and request.files['image']: #checking the method and file present or not
		app.logger.info(app.config['UPLOAD_FOLDER'])
		img = request.files['image']
		img_name = secure_filename(img.filename)
		print(img_name)
		create_new_folder(app.config['UPLOAD_FOLDER'])
		saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
		app.logger.info("saving {}".format(saved_path))
		img.save(saved_path)
		#return send_from_directory(app.config['UPLOAD_FOLDER'],img_name, as_attachment=True)
		counter=0
		distance=0
		img1 = cv2.imread("./uploads/"+img_name)
		face1 = getFace(img1) # Finding the threshold of the image
		print(type(face1))
		# for dirpath,dirname,filename in walk('./images/'):
		# 	print(filename)
		if os.path.getsize(pathPickle) > 0: 
			pickle_in=open(pathPickle,"rb")
			allImages=pickle.load(pickle_in)
		if allImages: 
			for keys in allImages: #iterating the pickle files with keys 
				img2=""
				img2 = allImages[keys][0] #assigning the threshold value to the variable img2
				distance = compare2face(face1, img2)
				threshold1 = 1   # set yourself to meet your requirement
				resultStr=""
				if(distance <= threshold1):
					pathImg=imageFuture+keys+'.'+(img_name.split('.'))[1] #creating the path of the image
					cv2.imwrite(pathImg, img1) # write the image in future folder
					
					addData(keys,face1,pathImg) #modifing the pickle folder with image path and the threashold of the image
					
					print("distance = "+str(distance)) #printing the distance between the two images
					
					resultStr={
					'Status ':'duplicate',
					'Face ID ' :keys,
					'Image Path ':allImages[keys][1]
					}
					print(resultStr)
					break
				else:
					counter=counter+1
		if(counter==len(allImages)): # check if the image is recognised or not.

			resultStr={
			'Status ':'No Match Find.'

			}
			#cv2.imwrite(imageDB+'FACE_'+id+'.'+(img_name.split('.'))[1], img1)
			print(resultStr)
			#addData('FACE_'+id+'.'+(img_name.split('.'))[1],face1)

		os.remove("./uploads/"+img_name) #removing the image from the server after checking the image.
		return jsonify(resultStr) #sending the result in json format
	else:
		return "Image Not Found"

# =========================MY Code Start ======================================
# 
# This Code add image to the database.
# Its Input are Client id and Client Image
@app.route('/add', methods = ['POST'])
def add_image():
	
	app.logger.info(PROJECT_HOME)
	if request.method == 'POST' and request.files['image']:
		app.logger.info(app.config['UPLOAD_FOLDER'])
		img = request.files['image']
		client_id=request.form['client_id']
		img_name = secure_filename(img.filename)
		print(img_name)
		create_new_folder(app.config['UPLOAD_FOLDER'])
		saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
		app.logger.info("saving {}".format(saved_path))
		img.save(saved_path)
		#return send_from_directory(app.config['UPLOAD_FOLDER'],img_name, as_attachment=True)
		counter=0
		distance=0
		img1 = cv2.imread("./uploads/"+img_name)
		face1 = getFace(img1)
		print(type(face1))
		# for dirpath,dirname,filename in walk('./images/'):
		# 	print(filename)
		pathImg=imageDB+client_id+'.'+(img_name.split('.'))[1]
		cv2.imwrite(pathImg, img1)
		addData(client_id,face1,pathImg)
		resultStr={
		'Status ':'success',
		'Face ID ':client_id,
		'Image Path ': pathImg
		}
		os.remove("./uploads/"+img_name)
		return jsonify(resultStr)
	else:
		return "Image Not Found "

@app.route('/del', methods = ['POST'])
def del_image():
	app.logger.info(PROJECT_HOME)
	if request.method == 'POST' and request.form['client_id']:
		client_id=request.form['client_id']
		res=delData(client_id)
		i=res.split('/')[-1]
		print(i)
		if os.path.exists(imageDB+i):
			os.remove(imageDB+i)
		if os.path.exists(imageFuture+i):
			os.remove(imageFuture+i)
		return 'Client removed'	

# =========================My Code Ends =======================================
if __name__ == '__main__':
    app.run(port=5003, debug=True)