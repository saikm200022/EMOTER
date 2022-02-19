import torch
import cv2
from facenet_pytorch import MTCNN

class SmileDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1 * 64 * 64, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 2)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

class FaceObjDetector:

    def __init__(self):
        self.network = MTCNN()
    
    def obj_detect(self, frame):
        boxes, pr, landmrks = self.network.detect(frame, landmarks = True)
        try:
            for box, p, landmark in zip(boxes, pr, landmrks):
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), thickness=3)
                cv2.circle(frame, tuple(landmark[0]), 5, (255, 0, 0), -1)
                cv2.circle(frame, tuple(landmark[1]), 5, (255, 0, 0), -1)
                cv2.circle(frame, tuple(landmark[2]), 5, (255, 0, 0), -1)
                cv2.circle(frame, tuple(landmark[3]), 5, (255, 0, 0), -1)
                cv2.circle(frame, tuple(landmark[4]), 5, (255, 0, 0), -1)
        except:
            pass

        return frame

model = SmileDetector()
# print(model(torch.rand(64, 3, 64, 64)).size())