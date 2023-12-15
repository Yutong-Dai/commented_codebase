
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl 
import torchmetrics
from torchmetrics import Metric
from pytorch_lightning.strategies import DeepSpeedStrategy

torch.set_float32_matmul_precision('medium')
class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += len(target)
        
    def compute(self):
        return self.correct.float() / self.total    

class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 10000)
        self.fc2 = nn.Linear(10000, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.my_accuracy = MyAccuracy()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _common_step(self, batch, batch_idx):
        x, y = batch 
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

        
    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        acc = self.accuracy(scores, y)
        myacc = self.my_accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({'train_loss':loss, 'train_accuracy':acc, 'train_myaccuracy':myacc, 'train_f1_score':f1_score}, 
                      on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss':loss, 'scores':scores, 'y':y}
    
    
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch 
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

# Initialize network
input_size = 784
num_classes = 10
model = NN(input_size=input_size, num_classes=num_classes)


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def prepare_data(self):
        # prepare the raw data
        datasets.MNIST(root=self.data_dir, train=True, download=True)
        datasets.MNIST(root=self.data_dir, train=False, download=True)
        
    def setup(self, stage):
        if stage == 'fit':
            entire_dataset = datasets.MNIST(root=self.data_dir, 
                                            train=True, 
                                            transform=transforms.ToTensor(), 
                                            download=False)
        
            self.train_ds, self.val_ds = random_split(entire_dataset, [50000, 10000], generator=torch.Generator().manual_seed(123))
        
        if stage == 'test':
            self.test_ds = datasets.MNIST(root=self.data_dir, 
                                    train=False, 
                                    transform=transforms.ToTensor(), 
                                    download=False)
    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test_ds, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=False)

dm = MnistDataModule(data_dir="/data/", batch_size=64, num_workers=32)

strategy = DeepSpeedStrategy(stage=2, logging_batch_size_per_gpu=64)

trainer = pl.Trainer(
    strategy=strategy,
    accelerator='gpu', devices=[0,1,2,3],
    min_epochs=1, max_epochs=3, precision='16-mixed')

trainer.fit(model, dm)




# disucssion on stepsizes and batch sizes
# https://github.com/Lightning-AI/pytorch-lightning/discussions/3706#discussioncomment-3960433