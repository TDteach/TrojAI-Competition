from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches


classes = [str(i) for i in range(100)]

def display_objdetect_image(image, boxes, labels, scores, score_threshold=0.7, out_name='out', score_top=None):
    if isinstance(image, str):
        image = Image.open(image)

    _, ax = plt.subplots(1, figsize=(12,9))
    image = np.array(image)
    ax.imshow(image)

    if score_top is not None:
        if score_top == 0:
            score_threshold = 0
        else:
            score_top = min(score_top, len(scores))
            order = np.argsort(scores)
            score_threshold = scores[order[-score_top]]

    # Showing boxes with score >= score_threshold
    for box, label, score in zip(boxes, labels, scores):
        if score >= score_threshold:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='b', facecolor='none')
            ax.annotate(classes[label] + ':' + str(np.round(score, 2)), (box[0], box[1]), color='w', fontsize=12)
            ax.add_patch(rect)

    plt.axis('off')
    plt.savefig(out_name+'.png', bbox_inches='tight')

