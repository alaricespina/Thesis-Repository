import cv2
import os

cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#video_capture = cv2.VideoCapture(0)
video_capture = cv2.VideoCapture('TestImagesandVideos/RecordingTest2.mp4')


while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()
    try:
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Display the resulting frame
        resized_frames = cv2.resize(frames, (1024,768))
        cv2.imshow('Video', resized_frames)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as E:
        print(f"Error : {E}")
        break

video_capture.release()
cv2.destroyAllWindows()