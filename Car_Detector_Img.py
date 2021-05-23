import cv2

#Our Image
img_file = 'Car Image.jpg'

# Our pre-trained car classifier
classifier_file = "Car_Dectector.xml"

#create opencv image
img = cv2.imread(img_file)

#convert img to black and white
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Car Classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#detecting Car
cars = car_tracker.detectMultiScale(black_n_white)

print(cars)

for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

#Show the Image
cv2.imshow('Soorya Car Detector', img)

# Don't Close
cv2.waitKey()

print("All Good!")