'''
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
'''

import time
import cv2
font = cv2.FONT_HERSHEY_COMPLEX

def show_webcam(text=None):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        img = cv2.flip(img, 1)
        text = text or str(time.time())
        if text:
            cv2.putText(img, text, (50, 50), font, 0.5, 
                        (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            import ipdb; ipdb.set_trace()
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == '__main__':
    show_webcam()
