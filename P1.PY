import cv2
import pygame

# Set up the music player
pygame.mixer.init()

# Load the songs based on emotions
happy_song = pygame.mixer.Sound("Songs/happy_song.wav")
sad_song = pygame.mixer.Sound("Songs/sad_song.wav")
angry_song = pygame.mixer.Sound("Songs/Aangry_song.wav")

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture video frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Check if any faces are detected
    if len(faces) > 0:
        # Get the first face detected
        (x, y, w, h) = faces[0]

        # Calculate the centroid of the face
        face_centroid = (x + w // 2, y + h // 2)

        # Determine the emotion based on the position of the face centroid
        if face_centroid[0] < frame.shape[1] // 3:
            emotion = "happy"
            pygame.mixer.music.stop()  # Stop any currently playing song
            happy_song.play()
        elif face_centroid[0] > 2 * frame.shape[1] // 3:
            emotion = "angry"
            pygame.mixer.music.stop()
            angry_song.play()
        else:
            emotion = "sad"
            pygame.mixer.music.stop()
            sad_song.play()


    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Exit the program if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
video_capture.release()
cv2.destroyAllWindows()
