import matplotlib.pyplot as plt
import numpy as np
import torch
import Config
import os

device = Config.DEVICE
figure_path = Config.FIGURE_PATH

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
        
        
# TODO: add more batch plot saving features        
def visualize_model(data, model, num_images=4, img_name = "plot_tmp"):

    model.eval()
    model.to(device)
    images_so_far = 0
    fig = plt.figure()
    dataloaders, class_names, dataset_sizes = data
    with torch.no_grad():
        # to check the first batch of data
        inputs, labels = next(iter(dataloaders['val']))
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title(f'predicted:{class_names[preds[j]]} [{class_names[labels[j]]}]')
            imshow(inputs.cpu().data[j])

                
        plt.savefig(os.path.join(figure_path, img_name))
        print('image saved to ', figure_path)
        plt.close(fig)

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))