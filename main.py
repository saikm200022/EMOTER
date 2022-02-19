'''
1) How to record and collect data from live video feed
2) Find an face emotions dataset
3) Train Image Classification model
4) Victory
'''


# import the opencv library
import cv2
import torch
from torchvision import transforms
from models import SmileDetector
from train import load_model, assess_accuracy
from PIL import Image

# define a video capture object
vid = cv2.VideoCapture(0)

model = SmileDetector()
model = load_model('RES8.th')
# print("ACCURACY: ", assess_accuracy(model))
while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    cv2.imshow('frame', frame)
    cv2.imwrite("frame.jpg", frame)
    image = Image.open("frame.jpg")
    transform = transforms.Compose([transforms.CenterCrop((64, 64)), transforms.Grayscale(1), transforms.ToTensor()])
    tensor = transform(image)
    label = model(tensor[None, :, :, :].float())
    print(label)
    if label.argmax() == 0:
        print("NOT SMILING *** : (")
    
    else:
        print("SMILING *** : )")

    # Display the resulting frame
      
    # the 'q' button is set as the
    # quitting button you may use any1
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()