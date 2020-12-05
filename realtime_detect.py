import torchvision
from torchvision import transforms
from utils import *
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

win_name='detect'

# download model
model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# model=torchvision.models.mobilenet_v2(pretrained=True)

# read image and transform it
trans=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0,0,0],[1,1,1])
])


model=model.eval()
model=model.to(device)
cam = cv2.VideoCapture(0)

# 80 classes in total
id2name=GetClass()
# prepare colors for every boxes
colors=np.random.randint(0, 255,(len(id2name),1,3))

with torch.no_grad():
    while True:
        ret, original = cam.read()
        # cv2.imshow(win_name, mat)
        mat = np.array(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        frame = trans(mat).unsqueeze(0)
        frame=frame.to(device)
        prediction = model(frame)
        try:
            labels = prediction[0]['labels']
            boxes = prediction[0]['boxes']
            scores = prediction[0]['scores']

            ShowPicResult(original, win_name, scores, labels, boxes,id2name,colors,False)
        except :
            print('something went wrong!')
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
