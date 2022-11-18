
import cv2

cam = cv2.VideoCapture(0)
cv2.namedWindow('python webcam screenshot app')

img_counter = 0
def main():
    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print('failed to grab frame')
            break
        
        cv2.imshow('test', frame)
        k  = cv2.waitKey(1)
        
        if k%256 == 27:
            print('escape hit, closing the app')
            break
        
        elif k%256  == 32:
            
            img_name = f'opencv_frame_{img_counter}'
            cv2.imwrite(img_name, frame)
            print('screenshot taken')
            img_counter += 1

    cam.release()
    cam.destoryAllWindows()


if __name__ == "__main__":
    main()