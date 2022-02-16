'''
1) How to record and collect data from live video feed
2) Find an face emotions dataset
3) Train Image Classification model
4) Victory
'''


# import the opencv library
import cv2
  
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any1
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()