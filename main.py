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
from models import SmileDetector, FaceObjDetector
from train import load_model, assess_accuracy
from PIL import Image

vid = cv2.VideoCapture(0)

model = SmileDetector()
model = load_model('RES8.th')
detector = FaceObjDetector()

while(True):
    ret, frame = vid.read()
    cv2.imwrite("frame.jpg", frame)
    image = Image.open("frame.jpg")
    transform = transforms.Compose([transforms.CenterCrop((64, 64)), transforms.Grayscale(1), transforms.ToTensor()])
    tensor = transform(image)
    label = model(tensor[None, :, :, :].float())
    text = "SMILING :)"
    if label.argmax() == 0:
        text = "NOT SMILING :("
    frame = detector.obj_detect(frame)
    display = cv2.putText(frame, text, (100, 100), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, thickness =3, color = (0,0,0))
    cv2.imshow("EMOTER", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()