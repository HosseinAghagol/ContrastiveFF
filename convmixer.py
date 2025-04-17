# https://openreview.net/forum?id=TVHS5Y4dNvM

import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
 nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
 nn.GELU(),
 nn.BatchNorm2d(dim),
 *[nn.Sequential(
 Residual(nn.Sequential(
 nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
 nn.GELU(),
 nn.BatchNorm2d(dim)
 )),
 nn.Conv2d(dim, dim, kernel_size=1),
 nn.GELU(),
 nn.BatchNorm2d(dim)
 ) for i in range(depth)],
 nn.AdaptiveAvgPool2d((1,1)),
 nn.Flatten(),
 nn.Linear(dim, n_classes)
)


class ConvMixer(nn.Module):
    def __init__(self,dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Sequential(nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
                                         nn.GELU(),
                                         nn.BatchNorm2d(dim),
                                         nn.Sequential(
                                                        Residual(nn.Sequential(
                                                                                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                                                                                nn.GELU(),
                                                                                nn.BatchNorm2d(dim)
                                                                              )
                                                                ),
                                                        nn.Conv2d(dim, dim, kernel_size=1),
                                                        nn.GELU(),
                                                        nn.BatchNorm2d(dim)
                                                      )
                                        )
                           )
        for l in range(1,depth):
            self.layers.append(nn.Sequential(
                                            Residual(
                                                     nn.Sequential(
                                                                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                                                                    nn.GELU(),
                                                                    nn.BatchNorm2d(dim)
                                                                  )
                                                    ),
                                            nn.Conv2d(dim, dim, kernel_size=1),
                                            nn.GELU(),
                                            nn.BatchNorm2d(dim)
                                            )
                              )
        self.classifier_head = nn.Sequential(
                                            nn.AdaptiveAvgPool2d((1,1)),
                                            nn.Flatten(),
                                            nn.Linear(dim, n_classes)
                                            )
    def forward(self, x):
        # Encoding
        for layer in self.layers:
            x = layer(x)
        # Pass through classification head
        x = self.classifier_head(x)
        return x