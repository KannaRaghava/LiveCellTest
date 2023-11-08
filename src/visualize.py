from data_preprocessing import train_data
from helper_functions import train_annotations, train_img_ids, cat_ids
import skimage.io as io
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def visualize():
    i = 89
    img_data = train_annotations.loadImgs(train_img_ids[i])
    ann_ids = train_annotations.getAnnIds(imgIds=img_data[0]['id'], catIds=cat_ids, iscrowd=0)
    anns = train_annotations.loadAnns(ann_ids)
    # mask = np.max(np.stack([train_annotations.annToMask(ann) * ann["category_id"] for ann in anns]), axis=0)
    # Create an empty array to store the stacked masks

    # Iterate through annotations and combine masks
    for ann in anns:
        mask = train_annotations.annToMask(ann) * ann["category_id"]

    img = io.imread(str(train_data + '/' + img_data[0]["file_name"]))
    img = img.astype('uint8')
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(TF.to_pil_image(img))

    plt.subplot(122)
    plt.imshow(mask)
    plt.show()