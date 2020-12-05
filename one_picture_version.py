import torchvision
from torchvision import transforms
from utils import *
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 80 classes in total
id2name=GetClass()
# prepare colors for every boxes
colors=np.random.randint(0, 255,(len(id2name),1,3))
win_name='detect'

# download model
model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# model=torchvision.models.mobilenet_v2(pretrained=True)

# read image and transform it
trans=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0,0,0],[1,1,1])
])

# change the path to the one you need
mat=cv2.imread('eg1.png')
mat=np.array(cv2.cvtColor(mat,cv2.COLOR_BGR2RGB))

test_example=trans(mat).unsqueeze(0)

# model predict
model=model.eval()

test_example=test_example.to(device)
model=model.to(device)
with torch.no_grad():
    prediction=model(test_example)

# grab the output to visualize result
labels=prediction[0]['labels']
boxes=prediction[0]['boxes']
scores=prediction[0]['scores']
print('start to draw picture')
ShowPicResult(mat,win_name,scores,labels,boxes,id2name,colors,True)
