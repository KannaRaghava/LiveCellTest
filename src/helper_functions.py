import torchvision
from pycocotools.coco import COCO
from data_preprocessing import train_data, test_data, train_json, test_json, LiveCellDataset



train_annotations = COCO(train_json)
test_annotations = COCO(test_json)

cat_ids = train_annotations.getCatIds(catIds=[ ], catNms=[], supNms=["cell", "cell"])
train_img_ids = []
for cat in cat_ids:
    train_img_ids.extend(train_annotations.getImgIds(catIds=cat))

train_img_ids = list(set(train_img_ids))
print(f"Number of training images: {len(train_img_ids)}")

test_img_ids = []
for cat in cat_ids:
    test_img_ids.extend(test_annotations.getImgIds(catIds=cat))

test_img_ids = list(set(test_img_ids))
print(f"Number of validation images: {len(test_img_ids)}")



# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)