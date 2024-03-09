CHAPTER 1 INTRODUCTION
1.1	PROBLEM DEFINITION
Visually impaired individuals face significant challenges in their daily lives, such as navigating unfamiliar environments, accessing printed materials, and effectively communicating with others. These challenges often lead to reduced independence and inclusion in an increasingly digital world.

1.2	OBJECTIVES
The primary objective of the EnVisionAI project is to enhance the daily life experiences of visually impaired individuals by providing essential features. While the project's web- based application is not directly used by visually impaired people, it serves as a crucial assistive tool. It aims to offer real-time object detection and recognition, improving spatial awareness and confidence. Additionally, it enables the conversion of images to text and speech, granting access to printed materials and information. The project also facilitates speech-to-text conversion, enhancing communication and note-taking, while text-to-speech conversion ensures accessibility to digital content and finally it also gives focus on the voice assistant feature. Ultimately, the EnVisionAI project strives to bridge the divide between visually impaired individuals and the digital world, fostering independence and inclusivity in an increasingly digital age.

1.3	OVERVIEW
EnVisionAI is a pioneering web-based application that brings four essential functionalities designed to enhance the daily experiences of visually impaired individuals. These critical functionalities encompass Object Detection, Image to Text/Speech Conversion, Speech to Text Conversion, Text to Speech and Voice Assistant. The project is dedicated to improving the independence and quality of life for visually impaired individuals by offering real-time object detection and recognition, converting images to text and speech, transforming speech to text, converting text to speech and Voice assistant to navigate between pages and answer simple questions. While the website is not intended for direct use by visually impaired people, it serves as a valuable assistive tool, bridging the gap between them and the digital world, and providing the necessary support for greater independence and inclusivity.
 
1.4	DOMAIN SPECIFICATION
The project operates within several domains, including Assistive Technology, Artificial Intelligence, Machine Learning, Computer Vision, and Web Development. These domains are leveraged to create a user-friendly web application that enables the features that are essential for visually impaired individuals to interact with their surroundings effectively.
 
CHAPTER 2 LITERATURE SURVEY

Bhasha Pydala, T. Pavan Kumar, K. Khaja Baseer proposed the “Smart_eye: A navigation and obstacle detection for visually impaired people through smart app”, where they present a proposed methodology for developing an Intellectual IoT System that aids in obstacle detection for visually impaired individuals. The system is designed to be cost- effective and efficient. The study aims to address existing research gaps and provide a simple and easy-to-use interface for visually impaired individuals. (Pydala et al. 2023)

Saad M. Darwish, Mohamed A. Salah and Adel A. Elzoghabi proposed the “Identifying indoor objects using neutrosophic reasoning for mobility assisting visually impaired people” where they present a framework for identifying indoor objects using neutrosophic reasoning to assist visually impaired people with mobility. The authors describe the technical details of their proposed system, including obstacle detection and classification modules, and provide experimental assessments using benchmarks and real data from visually impaired participants. The results demonstrate the dependability and robustness of the system, which was found to be user-friendly, lightweight, wearable, and unobtrusive. The authors suggest that their research could have a significant impact on the lives of visually impaired individuals by improving their mobility and independence. (Darwish et al. 2023).

Renas Rajab Asaad, Rasan Ismael Ali, Zeravan Arif Ali, Awaz Ahmad Shaban, proposed the “Image Processing with Python Libraries” ,Empowering computer vision to visually impaired people for satisfactory movements through Artificial Intelligence & Machine Learning with the help of nano-chemical concepts Section A-Research paper Empowering computer vision to visually impaired people for satisfactory movements through Artificial Intelligence & Machine Learning with the help of nano-chemical concepts, where they include developing object detection embedded with facial recognition software, emotion detection models, currency recognition software, web automation tools, and an emergency calling system. These technologies aim to enhance the understanding of the environment, identify emotions of individuals, assure transactions, automate web browsing, and provide assistance in panic situations for visually impaired individuals. (Asaad et al. 2023).
 
R.JeevaRekha, S.Divyabharathi, K.Mahalakshmi proposed the “Indoor and Outdoor Navigation Assistant System for Visually Impaired People using Yolo Technology”, which is about a system that uses Yolo Technology to help visually impaired people navigate indoors and outdoors. The system detects and analyses objects, allowing blind users to move around in unfamiliar environments with ease. The project presents a novel navigation device for visually impaired groups to help them reach the destination safely and efficiently. (JeevaRekha et al. 2023)

Vijayanarayanan A, Savithiri R, Lekha P, Abbirami R S proposed the “Image Processing Based on Optical Character Recognition with Text to Speech for Visually Impaired”, where they present an innovative technique that combines Optical Character Recognition (OCR) and Text to Speech Synthesizer (TTS) to help visually impaired individuals. The system uses a webcam to capture images of printed text, extract the text, and convert it into speech. The paper describes the design, implementation, and experimental results of the device, highlighting its potential to assist visually impaired people in reading newspapers, letters, and books independently. (Savithiri et al. 2023)

Sanjay Dutta, Sonu Dutta, Om Gupta, Shraddha Lone, Prof. Suvarna Phule propped the “Pisee: raspberry pi-based image to speech system for the visually impaired with blur detection” where they explore the use of image-to-speech conversion with blur detection for visually impaired individuals. It proposes a system using a Raspberry Pi and various libraries to convert images into speech and identify blurry images. (Dutta et al. 2023)

Hania Tarik, Shahzad Hassan, Rizwan Ali Naqvi, Saddaf Rubab, Usman Tariq, Monia Hamdi, Hela Elmannai, Ye Jin Kim, Jae‐Hyuk Cha proposed the “Empowering and conquering infirmity of visually impaired using AI‐technology equipped with object detection and real‐time voice feedback system in healthcare application”, where they present the development of ICANe, an intelligent cane for visually impaired individuals. The cane utilizes ultrasonic sensors and a camera to detect and identify obstacles, providing real-time voice feedback to the user for increased mobility and independence. (Tarik et al. 2023)

Ishwarya Sivakumar, Nishaali Meenakshisundaram, Ishwarya Ramesh, Shiloah Elizabeth, and Sunil Retmin Raj C, proposed the “VisBuddy - a smart wearable assistant for the visually challenged which focus on the Object Recognition and Speech Generation for Visually Impaired, the paper discusses a method called VisBuddy, which aims to assist
 
visually impaired individuals using object recognition and speech generation. The method utilizes the YOLOv5 image detection model and text-to-speech converters to identify objects in the environment and generate speech descriptions. The article compares two Python libraries for audio conversion and highlights the improved accuracy of the VisBuddy approach compared to existing methods. It also emphasizes the challenges faced by visually impaired individuals and the importance of addressing their needs. (Sivakumar et al. 2022)
 
CHAPTER 3 REQUIREMENT ANALYSIS

3.1	LIMITATIONS IN THE EXISTING SYSTEM
1.	Object Detection and Recognition: Many existing systems face challenges in accurate object detection, recognition, and classification. They may struggle to identify specific objects and provide meaningful information about them, hindering the user's understanding of their environment.
2.	OCR Difficulties for Text Documents: Optical Character Recognition (OCR) technology, while valuable, often encounters difficulties in accurately converting printed text into digital formats. This can result in errors and inaccuracies in the transcribed text, limiting the usability of the converted documents.
3.	Speech-to-Text Issues: Systems for converting spoken words to written text may suffer from inaccuracies, especially when dealing with multiple speakers, accents, or background noise. This can result in transcription errors that impact the quality of the converted text.
4.	Text-to-Speech Challenges: Text-to-speech technology can sometimes produce robotic or unnatural-sounding speech, making it less engaging and challenging to understand for visually impaired individuals. Moreover, these systems may struggle with context and intonation, affecting the overall quality of the spoken content.
These limitations underscore the need for more advanced and integrated solutions, like the EnVisionAI project, to address these challenges comprehensively and provide visually impaired individuals with more accurate and effective support in various aspects of their daily lives.
 
3.2	PROPOSED SYSTEM
The system utilizes advanced techniques, including the YOLO algorithm for object detection, the easyOCR library for image text recognition, and the Web Speech API for both speech-to-text, text-to-speech conversions and voice assistant, enabling a seamless and inclusive user experience for the visually impaired. This project aims to provide visually impaired individuals necessary features with a comprehensive web-based application to interact with and understand their surroundings. Each of the components mentioned has the potential to make a significant impact on the daily lives of visually impaired individuals. Here's a breakdown of each component:
1.	Object Detection:
a.	Object detection through a webcam or camera can be achieved using computer vision techniques, such as deep learning models.
b.	Providing information about detected objects through voice output is essential for accessibility.
2.	Voice Assistant:
a.	A voice assistant can greatly enhance user interaction and productivity.
b.	It can perform various tasks, such as web searches, toggling features, setting reminders, and more.
c.	Integrating a natural language understanding component can make the assistant more conversational and user-friendly.
3.	Image to Text/Speech:
a.	This feature can be beneficial for reading printed text from books, labels, or documents.
b.	Optical character recognition (OCR) technology can be used to extract text from images.
c.	Converting the extracted text to speech provides a means for visually impaired users to access printed information.
4.	Speech to Text:
a.	This functionality enables users to dictate text, which can be useful for note-taking, composing emails, or interacting with text-based applications.
b.	Accurate speech recognition is crucial to make this tool effective.
5.	Text to Speech:
a.	Converting text to speech allows users to consume digital content, including web articles, emails, or messages.
b.	Implementing high-quality text-to-speech synthesis can significantly enhance the user experience.

3.2.1	Input
The proposed system takes input from various sources:
a.	Object Detection: The system takes input from a camera (webcam or smartphone camera) to detect objects in the user's environment.
b.	Image to Text/Speech Conversion: Input is provided through images that contain text. The system recognizes and extracts text from these images.
c.	Speech to Text Conversion: The system listens to the user's spoken words and converts them into written text.
d.	Text to Speech Conversion: Users can input text directly into the system.
e.	Voice assistant: The system listens to the user's spoken words and try to respond to the users request.

3.2.2	Process
The system processes the input using the following modules:
a.	Object Detection: This module uses the YOLO algorithm to identify objects in the user's surroundings and provides auditory feedback to the user.
b.	Image to Text/Speech Conversion: The easyOCR library is employed to recognize and extract text from images. The system can present the recognized text in both text and speech form.
c.	Speech to Text Conversion: The Web Speech API is used to convert spoken words into text.
d.	Text to Speech: The system converts text input into audible speech using the Web Speech API.
e.	Voice assistant: The system can take the input of a user in the form of voice and can perform some basic tasks like searching on the web, switching the feature in the web app, etc using the Web Speech API.

3.2.3	Output
The system generates the following outputs for the user:
a.	Auditory descriptions of detected objects.
b.	Recognized text from images presented in both text and speech form.
c.	Converted text from speech input.
d.	Auditory playback of text input provided by the user.
e.	Auditory response reply for the voice commands.
 
3.3	SYSTEM REQUIREMENTS
3.3.1	HARDWARE REQUIREMENTS
For EnVisionAI, web application, which utilizes the inbuilt components of the system with sufficient processing power and memory for running the web application and handling real-time data processing, inbuilt webcam or camera for capturing real-time images for object detection and recognition, while the inbuilt microphone facilitates speech inputs and speech-to-text conversion. In addition, we will use system's inbuilt speakers or inbuilt headphones for auditory feedback, including text-to-speech conversion. An internet connection is necessary for real-time data processing and access to external resources like libraries and APIs.
Hardware Component	Requirement
Operating System	Windows 9 or later
Processor	Quad-core processor (e.g., Intel Core i5)
Memory (RAM)	8 GB or higher
Storage	SSD with sufficient storage for the OS and application
Webcam/Camera	Inbuilt webcam or external camera
Microphone	Inbuilt microphone or external microphone for speech input
Speakers/Headphones	Inbuilt speakers or headphones for auditory feedback and
text-to-speech output
Internet Connection	Broadband or high-speed internet connection for real-time
data processing and external resource access

3.3.2	SOFTWARE REQUIREMENTS
The software requirements for the system include Python for backend development, the Flask framework for server development, node.js for frontend development along with npm packages like react-speech-recognition, react-speech-kit, react-router-dom and various libraries and APIs such as YOLO for object detection, easyOCR for image text recognition, and the Web Speech API for speech-to-text and text-to-speech conversion.
The list of softwares to be installed in the system are:
1.	Frontend (React.js):
a.	Node.js and npm (Node Package Manager) - for managing frontend dependencies and running the development server of version 17.
2.	Backend (Python with Flask):
a.	Python - the programming language for backend development with version 3.9.
b.	Flask - a web framework for Python.
c.	Libraries for computer vision (e.g., numpy, pandas, pillow, TensorFlow, OpenCV, Keras, matplotlib).
d.	EasyOCR - for image to text/speech conversion.
e.	Web Speech API - for speech recognition and text-to-speech conversion.
3.	For YOLO Object Detection:
a.	YOLO weights -We will need to download and place the YOLO weights in the model_data directory.
 
CHAPTER 4 SYSTEM ARCHITECTURE
4.1	INTRODUCTION
The EnVisionAI system's architecture is the backbone of its functionality, orchestrating a seamless interaction between various components to serve the needs of visually impaired users. In this chapter, we will delve into the intricacies of the system's architecture, offering a comprehensive view of its design and organization. The architecture of the EnVisionAI system is a carefully crafted structure that harmonizes the critical modules, ensuring the delivery of its core functionalities. It encompasses a set of interconnected elements, each playing a pivotal role in the process of enhancing the lives of visually impaired individuals.

4.2	ARCHIITECTURE DIAGRAM
To provide a clear visual representation of the system's architecture, we will present an architecture diagram that illustrates the flow of data and interactions between the core components. This diagram will serve as a visual guide to help readers grasp how the different modules collaborate to deliver the desired functionalities.
 
4.3.	MODULES
The project focuses on the essential features required for virtual impaired people, focusing on the features, we have developed a website to explore the four features and for that we have developed the project based on these modules:
4.3.1	Object detection
The Object Detection module within the system harnesses the power of advanced object recognition, having undergone rigorous training using diverse datasets, including the extensive Common Objects in Context (COCO) dataset. With COCO's vast repository of over 200,000 labeled images spanning more than 80 object categories, this module offers precise identification and description of a wide array of objects within the user's environment. The utilization of such comprehensive training data enables the YOLO (You Only Look Once) algorithm to deliver real-time auditory feedback to visually impaired users, enhancing their spatial awareness and environmental understanding.
By incorporating the COCO dataset and other relevant datasets, this module ensures that visually impaired individuals can rely on an accurate and versatile object detection system to navigate and interact with the world around them effectively.
4.3.2	Image to Text / Speech Conversion
The Image to Text/Speech Conversion module is designed to convert images containing text into both readable text and audible speech. It utilizes the easyOCR library, which excels in Optical Character Recognition (OCR) tasks.
Users can provide images with printed text, and this module extracts and presents the text to the user in both textual and auditory formats. Notably, users can choose to hear the text content, effectively addressing the challenge of limited access to printed materials for visually impaired individuals.
4.3.3	Text to Speech Conversion
The Text to Speech Conversion module, equips visually impaired users with versatile tools for seamless interaction with digital content. This module, utilizing the capabilities of the Web Speech API, enables users to effortlessly convert written text into audible speech, providing a valuable means for accessing and comprehending textual information through auditory channels. Notably, users have the flexibility to customize their experience by setting the pitch and speed of the generated speech. Whether it be for reading articles, messages, or any written content, the Text to Speech Module enhances accessibility, offering visually impaired individuals a more inclusive and efficient user experience.
4.3.4	Speech to Text Conversion
The Speech to Text Conversion module, working in tandem with the overall system, provides visually impaired users with powerful tools for effortless interaction with digital content. Supported by the Web Speech API, the Speech to Text Conversion module allows users to seamlessly convert spoken language into written text, facilitating tasks like note- taking and verbal input. These capabilities significantly enhance the overall accessibility of textual information for visually impaired individuals, fostering a more inclusive and efficient user experience.
4.3.5	Voice assistant
The voice assistant module in my project serves as a pivotal component, empowering visually impaired individuals with a voice-controlled interface to interact with their surroundings and perform tasks. It allows users to issue voice commands, including web searches, application control, and information retrieval, while providing auditory feedback through text-to-speech conversion. This module is designed with natural language understanding capabilities and accessibility in mind, ensuring an inclusive and user-friendly experience. It seamlessly integrates with other application modules, such as object detection and image-to-text/speech, creating a holistic and empowering solution for visually impaired users, ultimately enhancing their independence and quality of life.
 
CHAPTER 5 IMPLEMENTATION DETAILS
The implementation of the "EnVisionAI" project involves developing a web-based application that leverages React.js for the frontend and Python with Flask for the backend. The application integrates computer vision algorithms, such as YOLO for object detection, and employs EasyOCR for image-to-text/speech conversion. Web Speech API is utilized for both speech recognition and text-to-speech conversion, enabling users to interact with the system through voice commands and receive auditory feedback.
SOURCE CODE
# Backend code – server.py
import os
from flask import Flask, flash, request, redirect, url_for, session, jsonify, send_file from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin import logging
from Models import * logging.basicConfig(level=logging.INFO) logger = logging.getLogger('HELLO WORLD') UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif']) app = Flask( name )
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/object-recognition', methods=['POST']) def ObjectRecognition():
print("========================Object Recognition=======================")
target=os.path.join(UPLOAD_FOLDER,'test_docs') if not os.path.isdir(target):
os.mkdir(target) logger.info("welcome to upload`") file = request.files['file']
filename = secure_filename(file.filename)
destination="/".join([target, 'input.jpg']) file.save(destination) session['uploadFilePath']=destination logger.info(destination)
# fun here
text=objRecog("uploads/test_docs/input.jpg", "uploads/test_docs/output.jpg") response=jsonify({'dataOutput':'output.jpg','dataInput':'input.jpg','text':text['labels']}) return response
@app.route('/image-to-text', methods=['POST']) def ImageToText():
print("========================OCR=======================")
target=os.path.join(UPLOAD_FOLDER,'test_docs') if not os.path.isdir(target):
os.mkdir(target) logger.info("welcome to upload`") file = request.files['file']
filename = secure_filename(file.filename) destination="/".join([target, 'input.jpg']) file.save(destination) session['uploadFilePath']=destination logger.info(destination)
text= Img2txt()
response=jsonify({'dataOutput':'output.jpg', 'dataInput':'input.jpg', 'text':text}) return response
@app.route('/display/<filename>') def display_image(filename):
#print('display_image filename: ' + filename) return send_file("uploads/test_docs/"+filename)
if    name 	== " main ": app.secret_key = os.urandom(24)
app.run(debug=True,host="0.0.0.0",use_reloader=False) # CORS(app, expose_headers='Authorization')
 
# Frontend code:
# index.js
mport React from "react";
import ReactDOM from "react-dom";
import { BrowserRouter } from "react-router-dom"; import "bootstrap/dist/css/bootstrap.css";
import "bootstrap/dist/js/bootstrap.bundle.js"; import App from "./App"; ReactDOM.render(
<BrowserRouter>
<App />
</BrowserRouter>, document.getElementById("root")
);


# App.js
import React from "react";
import { Route, Switch, Redirect } from "react-router-dom"; import Navbar from "./components/navbar";
import Home from "./components/home"; import Footer from "./components/footer"; import Aboutus from "./components/aboutus";
import ObjectDetection from "./components/objectDetection"; import ImageToText from "./components/imageToText"; import TextToSpeech from "./components/textToSpeech"; import SpeechToText from "./components/speechToText"; import PageNotFound from "./components/pageNotFound"; import "./App.css";

const App = () => (
<React.Fragment>
<div id="page-container">
<div id="content-wrap">
<Navbar />
<div>
 
<Switch>
<Route path="/object-detection" exact component={ObjectDetection} />
<Route path="/image-to-text" exact component={ImageToText} />
<Route path="/text-to-speech" exact component={TextToSpeech} />
<Route path="/speech-to-text" exact component={SpeechToText} />
<Route path="/about-us" exact component={Aboutus} />
<Route path="/" exact component={Home} />
<Route path="/page-not-found" exact component={PageNotFound} />
<Redirect to="/page-not-found" />
</Switch>
</div>
</div>
<Footer />
</div>
</React.Fragment>
);
export default App;


# Object detection
import os os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np import tensorflow as tf from .yolov3.utils import *
from .yolov3.configs import *
# image_path = "OBJ/image.jpg"
def detect_image(Yolo, image_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
original_image	= cv2.imread(image_path)
original_image	= cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) original_image	= cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) image_data = image_preprocess(np.copy(original_image), [input_size, input_size]) image_data = image_data[np.newaxis, ...].astype(np.float32)
 
if YOLO_FRAMEWORK == "tf":
pred_bbox = Yolo.predict(image_data) elif YOLO_FRAMEWORK == "trt":
batched_input = tf.constant(image_data) result = Yolo(batched_input)
pred_bbox = []
for key, value in result.items(): value = value.numpy() pred_bbox.append(value)

pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox] pred_bbox = tf.concat(pred_bbox, axis=0)
bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold) bboxes = nms(bboxes, iou_threshold, method='nms')
NUM_CLASS = read_class_names(CLASSES) labels=[]
scores=[] coordiantes=[]
print("=======================================================")
for i, bbox in enumerate(bboxes):
coor = np.array(bbox[:4], dtype=np.int32) score = bbox[4]
class_ind = int(bbox[5])
(x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3]) labels.append("{}".format(NUM_CLASS[class_ind])) scores.append(" {:.2f}".format(score)) coordiantes.append(coor)
# print(" {:.2f}".format(score), "{}".format(NUM_CLASS[class_ind]), (x1, y1), (x2,
y2))
print("======================================================")
image	=	draw_bbox(original_image,	bboxes,	CLASSES=CLASSES, rectangle_colors=rectangle_colors)
# CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))
 
if output_path != '': cv2.imwrite(output_path, image) if show:
# Show the image cv2.imshow("predicted image", image) # Load and hold the image cv2.waitKey(0)
# To close the window after the required kill value was provided cv2.destroyAllWindows()

return {"labels":labels, "scores": scores, "coordiantes": coordiantes} def objRecog(image_path, output_path):
yolo = Load_Yolo_model()
data = detect_image(yolo, image_path, output_path, input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0))
return data


# Image to text
import os os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import easyocr import cv2
from matplotlib import pyplot as plt import numpy as np
def Img2txt():
IMAGE_PATH = "./uploads/test_docs/input.jpg" IMAGE_Output_PATH = "./uploads/test_docs/output.jpg" reader = easyocr.Reader(['en','hi'])
result = reader.readtext(IMAGE_PATH) # print(result)
# print("Hello World") outputText="" img=cv2.imread(IMAGE_PATH) for detection in result:
top_left=tuple(detection[0][0]) bottom_right=tuple(detection[0][2])
 
text=detection[1] outputText+=text+" " # print(text, " ")
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.rectangle(img,[int(top_left[0]), int(top_left[1])], [int(bottom_right[0]), int(bottom_right[1])],(0,255,0),3)
img = cv2.putText(img,text,[int(top_left[0]), int(top_left[1])], font, 0.5, (0,0,0) ,2, cv2.LINE_AA)
cv2.imwrite(IMAGE_Output_PATH, img) return outputText
# plt.imshow(img) # plt.show()
 
CHAPTER 6
CONCLUSION AND FUTURE ENHANCEMENT
6.1	CONCLUSION
The "EnVisionAI" project is a pioneering web-based application that harnesses advanced technologies to enhance the lives of visually impaired individuals. Comprising four core modules, including Object Detection, Image to Text, Speech to Text, Text to Speech and Voice assistance this project empowers users in various ways. The Object Detection module, using the YOLO algorithm, provides real-time auditory feedback about the user's surroundings, while the Image to Text module extracts text from images, making printed material accessible. Speech to Text allows users to convert spoken words into written text via the Web Speech API, facilitating efficient data input. Meanwhile, the Text to Speech module employs the same API to transform text input into audible speech, ensuring accessibility to written content combining this two features voice assistant is integrated to navigate between pages and react to basic commands. This project not only employs cutting- edge technologies but also follows a well-structured timeline, with stages spanning from problem identification to module development and report preparation, culminating in a significant step toward improving the lives of the visually impaired.
6.2	FUTURE ENHANCEMENT
While EnVisionAI has made significant strides in addressing the challenges faced by visually impaired individuals, we acknowledge that technology is ever-evolving. As part of our commitment to continuous improvement, we have identified several areas for future enhancements:
1.	Advanced Object Recognition: We aim to further improve the accuracy and versatility of object detection by exploring state-of-the-art algorithms and expanding the dataset for training.
2.	Enhanced Text Recognition: Our future development will focus on overcoming challenges like varying lighting conditions and complex backgrounds, ensuring more robust text recognition from images.
3.	Natural Text to Speech: We plan to enhance the Text to Speech module by refining the speech synthesis to make it sound more natural and engaging. This will involve improvements in context and intonation.
 
REFERENCES

[1]	Bhasha Pydala, T. Pavan Kumar, K. Khaja Baseer, (2023) “SMART_EYE: A navigation and obstacle detection for visually impaired people through smart app”, in Journal of Applied Engineering and Technological Science, Vol 4(2): 992-1011.

[2]	Saad M. Darwish, Mohamed A. Salah and Adel A. Elzoghabi, (2023) “Identifying Indoor Objects Using Neutrosophic Reasoning for Mobility Assisting Visually Impaired People”, in Applied Sciences, MDPI, 13, 2150. https://doi.org/ 10.3390/app13042150.

[3]	Renas Rajab Asaad, Rasan Ismael Ali, Zeravan Arif Ali, Awaz Ahmad Shaban, (2023) “Image Processing with Python Libraries”, in Academic Journal of Nawroz University (AJNU),Vol.12,No.2,https://doi.org/10.25007/ajnu.v12 n2a1754.

[4]	R.JeevaRekha, S.Divyabharathi, K.Mahalakshmi, (2023) “Indoor and Outdoor Navigation Assistant System for Visually Impaired People using Yolo Technology”, inInternational Journal of New Innovations in Engineering and Technology, Volume 22 Issue 1, ISSN: 2319-6319.

[5]	Vijayanarayanan A, Savithiri R, Lekha P, Abbirami R S, (2023) “Image Processing Based on Optical Character Recognition with Text to Speech for Visually Impaired”, inJournal of Science, Computing and Engineering Research (JSCER),Volume-6, Issue- 4,DOI: https://doi.org/10.46379/jscer.2023.0604014

[6]	Sanjay Dutta, Sonu Dutta, Om Gupta, Shraddha Lone, Prof. Suvarna Phule, (2023) “PISEE: Raspberry pi-based image to speech system for the visually impaired with blur detection”, in International Research Journal of Modernization in Engineering Technology and Science, Volume:05,Issue:03, DOI : https://www.doi.org/10.56726/IRJMETS34522

[7]	Hania Tarik, Shahzad Hassan, Rizwan Ali Naqvi, Saddaf Rubab, Usman Tariq, Monia Hamdi, Hela Elmannai, Ye Jin Kim, Jae‐Hyuk Cha, (2023) “Empowering and conquering infirmity of visually impaired using AI‐technology equipped with object detection and real‐ time voice feedback system in healthcare application”, in CAAI Transactions on Intelligence Technology, DOI: 10.1049/cit2.12243
 
[8]	Ishwarya Sivakumar, Nishaali Meenakshisundaram, Ishwarya Ramesh , Shiloah Elizabeth, and Sunil Retmin Raj C, (2022) “VisBuddy - a smart wearable assistant for the visually challenged”, in arXiv:2108.07761v3.

[9]	https://github.com/JaidedAI/EasyOCR

[10]	https://developer.mozilla.org/enUS/docs/Web/API/Web_Speech_API/Using_ the_Web_Speech_API

[11]	https://pjreddie.com/media/files/yolov3.weights

[12]	https://flask.palletsprojects.com/en/2.0.x
