import torch
from torch import nn
from PIL import Image
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from natsort import natsorted
matplotlib.rcParams['text.usetex'] = True

N_ROT = 36

class tinyConv(nn.Module):
    def __init__(self):
        super(tinyConv, self).__init__()
        
        # 3x600x400
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 2), stride=(3,2))
        self.act1 = nn.LeakyReLU()
        # 16x200x200
        self.mp1 = nn.MaxPool2d(2, 2)
        # 16x100x100
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1, stride=2)
        # 32x50x50
        self.act2 = nn.LeakyReLU()
        self.mp2 = nn.MaxPool2d(2, 2)
        # 32x25x25
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2)
        # 64x12x12
        self.act3 = nn.LeakyReLU()
        self.mp3 = nn.MaxPool2d(2, 2)
        # 64x6x6
        
        self.flat = nn.Flatten()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.mp1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.mp2(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.mp3(x)

        x = self.flat(x)
        return x

m = tinyConv().eval()

def inference(img):
    x = np.array(img, dtype=float)  # HWC
    # normalize
    x -= x.mean()
    x /= x.std()
    x = np.moveaxis(x, -1, 0) # CHW
    x = x.reshape(1, *x.shape) # NCHW
    x = torch.Tensor(x)
    return m(x)


def calc_l2_norms(a, b):
    result = np.zeros((len(a), len(b)))
    # calculate all pairs of l2 norms
    for i in range(len(a)):
        for j in range(len(b)):
            result[i, j] = ((a[i] - b[j])**2).sum().sqrt()  # l2 norm
    return result


# minimize the main diagonal with cyclic permutations
def min_diag_cyclic_perm(A):
    diag = np.diag([1] * N_ROT)
    min_loss = np.inf
    
    for i in range(N_ROT):
        A_perm = np.roll(A, i, axis=1)   # positive = rotate reoriented frames CCW. (Should double check)
        loss = np.sum((diag * A_perm)**2)
        if loss < min_loss:
            min_loss = loss
            result = i
    return result


if __name__ == '__main__':
    # reference, 36 frames
    ref = []
    for path in natsorted(glob('mcs/imgs/ref/*.png')):
        ref.append(Image.open(path))
    ref_hash = [inference(x) for x in ref]

    reori = []
    for path in natsorted(glob('mcs/imgs/reori/*.png')):
        reori.append(Image.open(path))
    reori_hash = [inference(x) for x in reori]


    l2 = calc_l2_norms(ref_hash, reori_hash)
    plt.figure(dpi=150, figsize=(3,3))
    plt.imshow(l2, cmap='coolwarm')
    plt.title('$l^2$ norms, 36 rotations')
    plt.show()

