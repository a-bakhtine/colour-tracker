import cv2

# open default camera
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the frame
    cv2.imshow('Webcam feed', frame)

    # 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

# release
cam.release()
cv2.destroyAllWindows()