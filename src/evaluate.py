from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import torchvision

from model_eca import test_data_loader

# Set the model to evaluation mode
model.eval()

images, labels = next(iter(test_data_loader))

def view(images, labels, n=2, std=1, mean=0):
    figure = plt.figure(figsize=(15, 10))
    images = list(images)
    labels = list(labels)
    num_samples = min(n, len(images))  # Ensure you don't access out-of-range indices

    for i in range(num_samples):
        out = torchvision.utils.make_grid(images[i])
        inp = out.cpu().numpy().transpose((1, 2, 0))
        inp = np.array(std) * inp + np.array(mean)
        inp = np.clip(inp, 0, 1)
        ax = figure.add_subplot(2, 2, i + 1)
        ax.imshow(images[i].cpu().numpy().transpose((1, 2, 0)))
        if i < len(labels):
            l = labels[i]['boxes'].cpu().numpy()
            l[:, 2] = l[:, 2] - l[:, 0]
            l[:, 3] = l[:, 3] - l[:, 1]
            for j in range(len(l)):
                ax.add_patch(patches.Rectangle((l[j][0], l[j][1]), l[j][2], l[j][3], linewidth=1.5, edgecolor='r', facecolor='none'))

# Call the view function
view(images=images, labels=labels, n=2, std=1, mean=0)
