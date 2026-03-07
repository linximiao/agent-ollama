import torch
from torchvision import transforms as T
from torchvision.models.efficientnet import efficientnet_v2_m
from PIL import Image
import json
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('./models/dict.json', 'r', encoding='utf8') as f:
    dic = json.load(f)
def image_class(image_path:str) -> str:
    '''
    中草药分类
    参数：
        image_path (str):输入图片路径
    返回：
        str: 图片所属中草药类别
    '''
    net = torch.load('./models/efficientnet_v2_m.pth', weights_only=False)
    net = net.to(device)
    image = Image.open(image_path)
    transformer = T.Compose([
            T.Resize(384, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(384),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])
        ])
    image = transformer(image).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        p = net(image)
    _, p = torch.max(p, dim=1)
    return dic[f'{p.item()}']

if __name__ == '__main__':
    image_class('uploads\当归 (7).png')