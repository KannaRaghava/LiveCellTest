from data_preprocessing import train_data, test_data, train_json, test_json, LiveCellDataset
from helper_functions import get_transform, collate_fn

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torch.utils.data import DataLoader
import torchvision.transforms as T


class MySegmentationModel:
    def __init__(self,  num_classes=2, num_epochs=1):
        self.train_data = train_data
        self.train_json = train_json
        self.test_data = test_data
        self.test_json = test_json
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.train_batch_size = 1

        # Initialize datasets and data loaders
        self.my_dataset = self.create_dataset(self.train_data, self.train_json, get_transform())
        self.test_dataset = self.create_dataset(self.test_data, self.test_json, get_transform())

        self.data_loader = DataLoader(self.my_dataset, batch_size=self.train_batch_size, shuffle=True,
                                      num_workers=4, collate_fn=collate_fn)
        self.test_data_loader = DataLoader(self.test_dataset, batch_size=self.train_batch_size, shuffle=True,
                                           num_workers=4, collate_fn=collate_fn)

        # Create and initialize the model
        self.model = self.create_model(self.num_classes)
        self.model.to(self.device)

    def create_dataset(self, data_root, json_annotation, transforms):
        my_dataset = LiveCellDataset(root=data_root, annotation=json_annotation, transforms=transforms)
        return my_dataset

    def create_model(self, num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def train(self):
        len_dataloader = len(self.data_loader)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        for epoch in range(self.num_epochs):
            self.model.train()
            i = 0
            for imgs, annotations in self.data_loader:
                i += 1
                imgs = list(img.to(self.device) for img in imgs)
                annotations = [{k: v.to(self.device) for k, v in t.items()} for t in annotations]
                loss_dict = self.model(imgs, annotations)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                print(f'Epoch: {epoch}, Iteration: {i}/{len_dataloader}, Loss: {losses}')

        torch.save(self.model.state_dict(), 'maskrcnn_eca.pth')

    def get_transform(self):
        return T.Compose([
            T.ToTensor(),

        ])
