from time import time
import torch
import torchvision
import numpy as np

alexnet = torchvision.models.alexnet(pretrained=True).cuda()

x = torch.randn(32, 3, 256, 256).cuda()

# Inference
timings = []
for i in range(10):
    with torch.no_grad():
        t = time()
        alexnet(x)
        torch.cuda.synchronize()
        timings.append(time() - t)

timings = np.asarray(timings)
print("Min: ", np.min(timings), "s | Median: ", np.median(timings), "s | Max: ", np.max(timings), "s")
# Min:  0.018170595169067383 s | Median:  0.021053791046142578 s | Max:  0.05205416679382324 s

x = torch.randn(1, 3, 256, 256).cuda()

# Gradient
def loss_function():
    return alexnet(x).sum()

timings = []
for i in range(10):
    t = time()
    loss_function().backward()
    torch.cuda.synchronize()
    timings.append(time() - t)

timings = np.asarray(timings)
print("Min: ", np.min(timings), "s | Median: ", np.median(timings), "s | Max: ", np.max(timings), "s")
# Min:  0.07068514823913574 s | Median:  0.08000123500823975 s | Max:  0.11179089546203613 s