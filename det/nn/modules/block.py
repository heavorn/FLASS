import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, PConv


__all__ = (
    "R2NFN",
    "DFL",
    "SPPF",
    "FasterNetLayer",
    "C2f",
    "Bottleneck",
)

class R2NFN(nn.Module):
    """R2NFN"""

    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.e = (2 + n) / 2
        self.n = n + 2
        self.c = int(c2 * self.e) // 1
        self.g = self.c // self.n
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.m = nn.ModuleList(FasterNetLayer(self.g) for _ in range(self.n - 1))
        self.cv2 = Conv(self.c, c2, 1, 1)

    def forward(self, x):
        """Forward pass through R2NFN"""
        y = list(self.cv1(x).split(self.g, 1))
        r2n_layers = [y[0]]
        for i, r2n_layer in enumerate(self.m):
            x = y[i + 1] + r2n_layers[i] if i >= 1 else y[i + 1]
            r2n_layers.append(r2n_layer(x))
        return self.cv2(torch.cat(r2n_layers, 1))


class FasterNetLayer(nn.Module):
    """FasterNetLayer"""

    def __init__(self, c, e=1.0, n=4):
        super().__init__()
        self.c_ = int(c * e)
        self.cv1 = PConv(c, n)
        # self.cv2 = Conv(c, self.c_, 1, 1)
        self.cv2 = Conv(c, self.c_, 1, 1, act=nn.ReLU())
        # self.cv2 = Conv(c, self.c_, 1, 1, act=nn.GELU())
        self.cv3 = nn.Conv2d(self.c_, c, 1, 1, bias=False)

    def forward(self, x):
        """Forward pass through FasterNetLayer"""
        return x + self.cv3(self.cv2(self.cv1(x)))


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(
            b, 4, a
        )
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))



class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self, c1, c2, n=1, shortcut=False, g=1, e=0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )
        # self.m = nn.ModuleList(MSBlockLayer(self.c, self.c) for _ in range(n))
        # self.m = nn.ModuleList(FasterNetLayer(self.c) for _ in range(n))
        # self.m = nn.ModuleList(GSBottleneck(self.c, self.c, 1, 1) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
    
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5
    ):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))