import cv2
import numpy as np 

def main():
    # open default camera
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    # colour presets (HSV)
    colours = {
        "blue": {
            "min": np.array([100, 150, 50]),
            "max": np.array([140, 255, 255])
        },
        "green": {
            "min": np.array([40, 70, 50]),
            "max": np.array([80, 255, 255])
        },
        # red wraps around 
        "red": {
            "min": np.array([0, 150, 160]),
            "max": np.array([10, 255, 255]),
            "min2": np.array([170, 150, 170]),
            "max2": np.array([180, 255, 255])
        },
        "yellow": {
            "min": np.array([20, 180, 160]),
            "max": np.array([35, 255, 255])
        }
    }

    # start with blue
    current_colour = "blue"
        
    min_blue = np.array([100, 150, 50])
    max_blue = np.array([140, 255, 255])

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # frame BGR -> HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        colour_info = colours[current_colour]
        
        # red needs 2 ranges
        if current_colour == "red":
            mask1 = cv2.inRange(hsv, colour_info["min"], colour_info["max"])
            mask2 = cv2.inRange(hsv, colour_info["min2"], colour_info["max2"])
            # combine masks
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, colour_info["min"], colour_info["max"])
        
        # cleanup to remove noise
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)

        # keep only coloured parts of OG frame
        colour_only = cv2.bitwise_and(frame, frame, mask=mask)

        # make a box around the blue area
        # coords of all pixels in mask
        ys, xs = np.where(mask > 0)

        if len(xs) > 0 and len(ys) > 0:
            # find min and max
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()


            area = (x_max - x_min) * (y_max - y_min)
            frame_area = frame.shape[0] * frame.shape[1]
            
            # ignore noise and too big boxes
            if 500 < area < (0.5 * frame_area):
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

        # show which colour is tracked
        text = f"Tracking: {current_colour.upper()}"
        cv2.putText(
            frame,
            text,
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, # font size
            (255,255,255),
            2 
        )

        # show instructions
        text = "B:Blue | G:Green | R:Red | Y:Yellow | Q:Quit"
        cv2.putText(
            frame,
            text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,255),
            1
        )

        # display frames
        cv2.imshow('Webcam feed', frame)
        cv2.imshow("Colour feed", colour_only)

        # key press
        key = cv2.waitKey(1)

        # exit
        if key == ord('q') or key == ord('Q'):
            break
        # colour switches
        elif key == ord('b') or key == ord('B'):
            current_colour = "blue"
        elif key == ord('g') or key == ord('G'):
            current_colour = "green"
        elif key == ord('r') or key == ord('R'):
            current_colour = "red"
        elif key == ord('y') or key == ord('Y'):
            current_colour = "yellow"

    # release
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()