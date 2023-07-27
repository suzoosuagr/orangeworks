import matplotlib.pyplot as plt

def disp_inline_images(images:list, titles:list, *args, **kwargs) -> None:

    fig, axs = plt.subplots(1, len(images), *args, **kwargs)
    cmap = 'viridis'
    if len(images[0].shape) == 2:
        cmap = 'gray'
    if len(images) == 1:
        axs.imshow(images[0], cmap=cmap)
        axs.title.set_text(titles[0])
    else:
        for i in range(len(images)):
            axs[i].imshow(images[i],  cmap=cmap)
            axs[i].title.set_text(titles[i])
    plt.show()
    plt.close('all')