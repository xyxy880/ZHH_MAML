import cv2
import numpy as np

def img_show(images):
    imgs=[]
    for i in range(len(images)):
        x = images[i]
        # x = x.data.cpu().numpy()#用了gpu就加这行
        x = np.asarray(x)
        x = np.transpose(x, (1, 2, 0))#转成 chw 才能用cvt函数
        # cv2.imwrite("./test.png", x)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        img = np.transpose(x, (2, 0, 1))
        imgs.append(img)

    return imgs

