import cv2

#Our Video
video = cv2.VideoCapture('Pedestrians.mp4')

# Our pre-trained car classifier
classifier_file = "Pedestrian_Dectector.xml"

#Car Classifier
Pedestrian_tracker = cv2.CascadeClassifier(classifier_file)

#run forever unitl video stops
while True:

    #read current frame
    (read_successful, frame) = video.read()

    #convert video from black and white
    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detecting Pedestrain
    pedestrians = Pedestrian_tracker.detectMultiScale(grayscaled_frame)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    #Show the Video
    cv2.imshow('Soorya Pedestrians Detector', frame)

    cv2.waitKey(1)
