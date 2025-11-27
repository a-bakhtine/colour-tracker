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

        # make a box around the blue area
        # coords of all pixels in mask
        ys, xs = np.where(mask > 0)

        if len(xs) > 0 and len(ys) > 0:
            # find min and max
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()


            # ignore noise (if area LT 500 no box)
            if (x_max - x_min) * (y_max - y_min) > 500:
                # draw tracking rectangle
                cv2.rectangle(
                    frame,
                    (x_min, y_min), # top left
                    (x_max, y_max), # bottom right
                    (0,255,0), # colour (BGR)
                    2 # thickness
                )
                
                # draw dot in center
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2

                cv2.circle(
                    frame,
                    (center_x, center_y), # center of the box
                    5, # radius
                    (0, 0, 255), # red
                    -1 # filled circle
                )



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