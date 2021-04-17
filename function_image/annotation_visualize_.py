import json
import pathlib

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# =======================
# visualize annotation
# =======================

def read_annotation(CFG, data_id):
    file_path = CFG.ROOT_PATH + f"{CFG.annotation_path}/{data_id}.json"
    return read_json(file_path)

def read_image(CFG, data_id, gray=False):
    file_path = CFG.ROOT_PATH + f"{CFG.image_path}/{data_id}.png"
    print(file_path)
    if gray:
        img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    return img

def read_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

def show_madri(data_ids, title=""):
    n_sample = 16
    fig, axes = plt.subplots(4, 4, figsize=(20, 20), constrained_layout=True, facecolor="lightgray")
    fig.suptitle(title, fontsize=20)
    axes = axes.flatten()
    for i in range(n_sample):
        data_id = data_ids[i]
        image = read_image(CFG, data_id)
        axes[i].set_title(f"{data_id}")
        axes[i].grid(False)
        axes[i].imshow(image)

def is_polygon(CFG, code):
    return code <= CFG.polygon

def show_annoteted_madori(CFG, n_sample = 10, title=""):
    cmap = plt.get_cmap("tab10")
    ncol = 2
    nrow = (n_sample + 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(10*ncol, 10*nrow), constrained_layout=True, facecolor="lightgray")
    fig.suptitle(title, fontsize=20)
    axes = axes.flatten()

    for i, data_id in enumerate(train_ids[:n_sample]):
        # Use grayscale.
        image = read_image(CFG, data_id, gray=True)
        annot = read_annotation(CFG, data_id)
        ax = axes[i]
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{data_id}: {image.shape[0]}x{image.shape[1]}")
        for label in annot["labels"]:
            for item in annot["labels"][label]:
                label_code = CFG.LABEL_CODE[label]
                pts = np.array(item)
                if is_polygon(CFG, label_code):
                    ax.fill(pts[:, 0], pts[:, 1], alpha=0.65, c=cmap.colors[label_code])
                    ax.text(pts[:, 0].mean(), pts[:, 1].mean(), label_code)
                else:
                    xmin, ymin, xmax, ymax = pts
                    w = xmax - xmin
                    h = ymax - ymin
                    box = mpl.patches.Rectangle(
                        (xmin, ymin), w, h, alpha=1.0, color=cmap.colors[label_code], fill=False, lw=5
                    )
                    ax.add_patch(box)
                    ax.text(
                        0.5 * (xmax + xmin),
                        0.5 * (ymax + ymin),
                        label_code,
                        horizontalalignment="center",
                        verticalalignment="center",
                    )

                   
# ====================
# usage
# ====================
#CFG.LABEL_CODE = {"LDK": 0, "廊下": 1, "浴室": 2, "洋室": 3,  # polygon  "引戸": 4, "折戸": 5, "開戸": 6  # box}
#CFG.polygon = 3
#CFG.ROOT_PATH = '/content/'
#CFG.annotation_path = 'train_annotations'
#CFG.image_path = 'train_images'
#train_ids = [p.stem for p in (pathlib.Path("/content/") / "train_images").glob("*.png")]
#show_madri(train_ids, "TRAIN")
#show_annoteted_madori(title="ANNOTATED")
