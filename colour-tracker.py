import cv2
import numpy as np 

def main():
    # open default camera
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    #HSV range of what's considered blue
    min_blue = np.array([100, 150, 50])
    max_blue = np.array([140, 255, 255])

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # frame BGR -> HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # mask (white where blue, black everywhere else)
        mask = cv2.inRange(hsv, min_blue, max_blue)

        # keep only blue parts of OG frame
        blue_only = cv2.bitwise_and(frame, frame, mask=mask)

        # display frames
        cv2.imshow('Webcam feed', frame)
        cv2.imshow("Blue feed", blue_only)

        # 'q' to exit
        if cv2.waitKey(1) == ord('q'):
            break3

    # release
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()