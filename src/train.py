import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from data_preprocessing import train_data, test_data, train_json, test_json, LiveCellDataset
from src.model_eca import MySegmentationModel
from src.model_se import SEModelTrainer
from src.helper_functions import get_transform

class ModelTrainer:
    def __init__(self, train_data, train_json, num_classes=2, num_epochs=1, model_type='fasterrcnn'):
        self.train_data = train_data
        self.train_json = train_json
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.model_type = model_type

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.train_batch_size = 1
        self.data_loader = DataLoader(self.create_dataset(), batch_size=self.train_batch_size, shuffle=True, num_workers=4, collate_fn=self.collate_fn)

        if self.model_type == 'fasterrcnn':
            self.model = MySegmentationModel(self.num_classes)
        elif self.model_type == 'se':
            self.model = SEModelTrainer()

        self.model.to(self.device)

    def create_dataset(self):
        my_dataset = LiveCellDataset(root=self.train_data, annotation=self.train_json, transforms=self.get_transform())
        return my_dataset

    def create_se_model(self):
        model = get_model_instance_segmentation(self.num_classes)
        self.modify_resnet_backbone(model.backbone.body)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        return model

    def modify_resnet_backbone(self, resnet_backbone):
        resnet_backbone.layer2[3].conv2 = SEBlock(in_channels=resnet_backbone.layer2[3].conv2.in_channels)

    def collate_fn(self, batch):
        return tuple(zip(*batch))

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

        if self.model_type == 'fasterrcnn':
            torch.save(self.model.state_dict(), 'trained_models/maskrcnn_eca.pth')
        elif self.model_type == 'se':
            torch.save(self.model.state_dict(), 'trained_models/maskrcnn_se.pth')

if __name__ == "__main__":
    # Train the first model (FastRCNN)
    model_trainer = ModelTrainer(train_data, train_json, num_classes=2, num_epochs=1, model_type='fasterrcnn')
    model_trainer.train()

    # Train the second model (SE)
    se_model_trainer = ModelTrainer(train_data, train_json, num_classes=2, num_epochs=1, model_type='se')
    se_model_trainer.train()
