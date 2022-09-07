

*Go, change the world*

**RV College of**

**Engineering**

**Artificial Intelligence and Machine**

**Learning**

**Project Presentation**

**Department of Computer Science,**

**R V College of Engineering, Bengaluru.**





*Go, change the world*

**RV College of**

**Engineering**

**Verification of faces in Near-Infrared Images**

Athreya V Shet - 1RV19CS029

Chinmay B S - 1RV19CS041

**Faculty In charge**

Dr. Shobha G

Professor

<img src="images/Picture1.png" alt="drawing" width="200"/>



*Identification of Problem*

*and Analysis*

*Go, change the world*

**RV College of**

**Engineering**

● Face recognition from image or video is a popular topic in biometrics research. Many public

places usually have surveillance cameras for video capture and these cameras have their

significant value for security purpose.

● I R cameras are the ones, where the camera collects footage even in low light. This feature adds

a considerable advantage in recognizing faces in less lux of light intensity. The image obtained

by the I R camera is de facto Gray-scale thus has considerable advantage in training of

machine learning models.





*Identification of Problem*

*and Analysis*

*Go, change the world*

**RV College of**

**Engineering**

**Near Infrared Images:**

● Near Infrared energy is the portion of the electromagnetic spectrum just past the Red segment of visible

light. Infrared energy is made up of longer wavelengths than the Red, Green or Blue light so our eyes

aren’t capable of seeing its effects.

● Fortunately some modern digital cameras are sensitive to this longer wavelength and can record its

stunning properties.





*Go, change the world*

**RV College of**

**Engineering**

*Literature Study*

**EXISTING SYSTEMS**

**1. Face detection techniques.**

● Face detection is classified into image base and feature base wherein image based includes neural network &

statistical approaches( PCA,SVM etc). one of the best existing models for face detection is YOLO-face, whose

architecture is based on YOLOv3. It has the same face detection speed as YOLOv3 but it solves the problem of

varying face scales. OpenCV has a haar classifier which is used for face detection.





*Go, change the world*

**RV College of**

**Engineering**

*Literature Study*

**2. Face Recognition techniques**

*Flow diagram of the stages involved in a face recognition system*

● Algorithms for face recognition include PCA(Principal component analysis), fisherface, SVD- based recognition etc.

● The PCA algorithm is based on using eigenvectors. We hence don’t need to design classifiers for these models as Euclidean distance can

be used to recognize faces because the generated vectors are not correlated to each other. accuracy of PCA alone was found to be 91%.

● The fisherface method is based on the ratio of the similarity in the faces of two or more people for the basis of comparison.

● SVD-based face recognition is a method according to which a singular value decomposition for projection and recognition is used. It is

basically a 2D-PCA algorithm. Another method is LDA-linear discriminant analysis which has an accuracy of 94%.

● Deep learning approaches are also used in classification of faces.





*Go, change the world*

**RV College of**

**Engineering**

Design Methodology





Planning of project work

*Go, change the world*

**RV College of**

**Engineering**

**Various stages planned and flow of the tasks**

\- In first step we decided to create dataset using IR camera in low light conditions(near infrared

images)

\- In next step we processed created dataset using face detection algorithms like Haarcascade

and Yoloface

\- In this stage we used VGG face recognition algorithm to recognize detected faces. We used

dataset created using face detection algorithm to train and test face recognition algorithm.

\- In next step we added one more face detection algorithm and wrote comparison on these face

detection algorithms.

\- In final step we tested our face recognition algorithm using a frame with single face, multiple

faces, and different orientations.





Planning of project work

*Go, change the world*

**RV College of**

**Engineering**

**Dataset Created using live CCTV camera**

***Figure 6: Sample faces from the dataset created (15 different faces***

***in the dataset)***





Planning of project work

*Go, change the world*

**RV College of**

**Engineering**

**Detection algorithms used in the project**

• **Haar-cascade** module, is available along with the OpenCV module. The detection was carried out using the

haar frontal face xml, which draws a square bounding box after finding faces in images, using haar features and

selecting best features using Adaboost.

• **Yolo-face** based on YOLOv3 was derived to improve the performance for face detection. The approach in

yolo includes using anchor boxes more appropriate for face detection and a more precise regression loss

function. The improved detector significantly increased accuracy while remaining fast detection speed. [8]

experimented yolo-face on the WIDER FACE and the FDDB datasets and showed that the improved algorithm

outperforms YOLO and its varieties.

• **MTCNN** or Multi-Task Cascaded Convolutional Neural Networks composed of three convolutional networks

(P-Net, R-Net, and O-net) is very robust and can outperform many face-detection benchmarks but manages to

work in real-time. Composed of 3 neural networks, each with multiple layers allows for higher precision [9].

The paper claims 16fps on a 2.60GHz CPU and higher in other higher specification in the mentioned un-

optimized MATLAB code.





Project Demonstration and Result Analysis

*Go, change the world*

**RV College of**

**Engineering**

Face Detection Techniques Comparison

**MTCNN**

Total Actual

frames frames with

**Haar-Cascade**

**YOLO-Face**

Total Actual

frames with frames frames with

Total

frames

detected

Actual

**Dataset No**

detected

faces

faces

detected

faces

**1**

**2**

259

62

259

62

282

57

282

57

280

63

280

63

**3**

226

193

302

225

270

292

88

226

174

284

220

239

288

83

187

197

297

232

280

292

111

157

270

279

254

94

185

196

297

232

280

292

111

157

268

279

254

94

240

234

300

233

282

300

135

160

273

296

278

101

300

240

234

300

233

282

300

135

160

273

296

266

99

**4**

**5**

**6**

**7**

**8**

**9**

**10**

**11**

**12**

**13**

**14**

**15**

110

264

281

249

66

90

263

277

208

46

187

132

289

289

297

**Table 2: Detection algorithms result on the dataset created**





Planning of project work

*Go, change the world*

**RV College of**

**Engineering**

**Using Transfer Learning to recognize custom faces.**

• **Transfer learning** is a machine learning method where a model developed for a task is reused as the

starting point for a model on a second task. Transfer learning reduces the amount of time that you need to

spend on training.

• In general CNN, models for image classification can be divided into two parts:

***Feature extraction***: the aim of this part is to find the features in the image

***Classification***: the aim of this part is to use the various features extracted in the previous part and

classify the image to the desired classes.





Planning of project work

*Go, change the world*

**RV College of**

**Engineering**

**VGG-FACE Recognition Module**

• **VGGFace** refers to a series of models developed for face recognition. It was developed by the Visual

Geometry Group (hence its VGG name) at the University of Oxford.

• The models were trained on a dataset comprised mainly of celebrities, public figures, actors, and politicians.

Their names were extracted from the Internet Movie Data Base (IMDB) celebrity list based on their gender,

popularity, pose, illumination, ethnicity, and profession (actors, athletes, politicians).





Planning of project work

*Go, change the world*

**RV College of**

**Engineering**

**Face classification Model**

• A customized classifier, softmax regressor

was built with 3 layers, containing first

layer with 100 units and tanh activation

function,

• second layer with 10 units and tanh

activation function and

• third layer with 6-15 units for required no

of outputs, for each person followed by a

layer with softmax activation function.





**RV College of**

**Engineering**

Face Recognition Results

*Go, change the world*

● With the detection and recognition methods mentioned above, we trained our VGG-face derived

classifier to recognize faces present in the frame.

● The final model involved both detection and recognition in a single pass, and the algorithm worked

well in real time near infrared video and below is the result as a snapshot of recognition, where

multiple faces in the video were detected and correctly labeled according to the dataset.









Conclusion & future scope

*Go, change the world*

**RV College of**

**Engineering**

● Face detection and recognition in Near infrared images can have a serious advantage in cctv

video auditing especially in night time. As, near infrared image is in gray-scale it reduces the

bulkiness of algorithm performance as that of RGB images. The technique proposed involves

usage of detection algorithm with VGG face derived softmax classifier, thus resulting in

effective recognition method.

● From these detection algorithms we can conclude that Haar-cascade is fast compared to other

algorithms but does not detect all faces and also it detects wrong faces. So, one way is to use

MTCNN or YOLO face for training because they detect faces accurately and use Haar

Cascade for real time face detection as it is fast.

● We can integrate this project according to the need, for example as part of an intruder

detection systems or Automatic access logs for secure facilities. we can also purpose this to be

used in attendance gathering systems. we can use his as part of survellience network for the

security agencies. we can further develop the systems





References

*Go, change the world*

**RV College of**

**Engineering**

Research Papers:

\1. S. L. a. N. I. N. Jamil, ""Face recognition using neural networks,"," Proceedings. IEEE International Multi

.

Topic Conference, 2001. IEEE INMIC 2001. Technology for the 21st Century., , pp. pp. 277-281, 2001.

2 N. R. B. a. S. Kuwelkar, " "Real-time implementation of face recognition system,"," International

Conference on Computing Methodologies and Communication (ICCMC), pp. pp. 249-255, 2017

3 W. e. a. Chen, " “YOLO-face: a real-time face detector.”," The Visual Computer 37 , pp. 805-813., (2020).

Web links:

[1.](https://github.com/serengil/deepface)[ ](https://github.com/serengil/deepface)<https://github.com/serengil/deepface>

[2.](https://www.analyticsvidhya.com/blog/2022/04/object-detection-using-haar-cascade-opencv/)[ ](https://www.analyticsvidhya.com/blog/2022/04/object-detection-using-haar-cascade-opencv/)<https://www.analyticsvidhya.com/blog/2022/04/object-detection-using-haar-cascade-opencv/>

[3.](https://medium.com/analytics-vidhya/face-recognition-with-vgg-face-in-keras-96e6bc1951d5)[ ](https://medium.com/analytics-vidhya/face-recognition-with-vgg-face-in-keras-96e6bc1951d5)<https://medium.com/analytics-vidhya/face-recognition-with-vgg-face-in-keras-96e6bc1951d5>

[4.](https://pypi.org/project/yoloface/)[ ](https://pypi.org/project/yoloface/)<https://pypi.org/project/yoloface/>

[5.](https://towardsdatascience.com/face-detection-using-mtcnn-a-guide-for-face-extraction-with-a-focus-on-speed-c6d59f82d49)[ ](https://towardsdatascience.com/face-detection-using-mtcnn-a-guide-for-face-extraction-with-a-focus-on-speed-c6d59f82d49)[https://towardsdatascience.com/face-detection-using-mtcnn-a-guide-for-face-extraction-with-a-focus-on-](https://towardsdatascience.com/face-detection-using-mtcnn-a-guide-for-face-extraction-with-a-focus-on-speed-c6d59f82d49)

[speed-c6d59f82d49](https://towardsdatascience.com/face-detection-using-mtcnn-a-guide-for-face-extraction-with-a-focus-on-speed-c6d59f82d49)

