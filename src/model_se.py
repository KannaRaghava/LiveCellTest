from model_eca import data_loader
from data_preprocessing import train_data, test_data, train_json, test_json, LiveCellDataset

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.se(x)
        return x * out

class SEModelTrainer:
    def __init__(self, num_classes=2, num_epochs=1):
        self.train_data = train_data
        self.train_json = train_json
        self.num_classes = num_classes
        self.num_epochs = num_epochs

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.train_batch_size = 1
        self.data_loader = DataLoader(self.create_dataset(), batch_size=self.train_batch_size, shuffle=True, num_workers=4, collate_fn=self.collate_fn)

        self.model = self.create_se_model()
        self.model.to(self.device)

    def create_dataset(self):
        my_dataset = LiveCellDataset(root=self.train_data, annotation=self.train_json, transforms=self.get_transform())
        return my_dataset

    def create_se_model(self):
        model = fasterrcnn_resnet50_fpn(pretrained=False)
        self.modify_resnet_backbone(model.backbone.body)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        return model

    def modify_resnet_backbone(self, resnet_backbone):
        resnet_backbone.layer2[3].conv2 = SEBlock(in_channels=resnet_backbone.layer2[3].conv2.in_channels)

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def get_transform(self):
        # Define your data transformation here
        # Example:
        return T.Compose([
            T.ToTensor(),
            # Add more transformations as needed
        ])

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

        torch.save(self.model.state_dict(), 'maskrcnn_se.pth')

# Example usage:
# se_model_trainer = SEModelTrainer(train_data, train_json, num_classes=2, num_epochs=1)
# se_model_trainer.train()
