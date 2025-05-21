import cv2
import time
from mc.connexio import connecta

mc = connecta()
mc.postToChat("Detector de moviment actiu.")

cap = cv2.VideoCapture(0)
_, frame1 = cap.read()
_, frame2 = cap.read()

while True:
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        print("🎯 MOVIMENT detectat!")
        mc.postToChat("MOVIMENT detectat!")
        time.sleep(2)  # Evita repeticions constants

    frame1 = frame2
    _, frame2 = cap.read()
    cv2.imshow("Detector de moviment", frame2)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
