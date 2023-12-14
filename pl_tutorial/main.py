# %%
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
import pytorch_lightning as pl 


# %% [markdown]
# some useful methods in pl.LightningModule
# * print
# * log
# * forward
# * training_step
# * training_step_end
# * training_epoch_end
# * validation_step
# * test_step
# * predict_step
# * configure_optimizers

# %%
class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

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
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
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

# %%
# Initialize network
input_size = 784
num_classes = 10
model = NN(input_size=input_size, num_classes=num_classes)


# %%

batch_size = 64
# Load Data
entire_dataset = datasets.MNIST(
    root="/data/", train=True, transform=transforms.ToTensor(), download=True
)
train_ds, val_ds = random_split(entire_dataset, [50000, 10000])
test_ds = datasets.MNIST(
    root="/data/", train=False, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)

# %%
trainer = pl.Trainer(accelerator='gpu', devices=[0,1],
                     min_epochs=1, max_epochs=3, precision='16-mixed',
                     )


# %%
trainer.fit(model, train_loader, val_loader)

# %%
trainer.validate(model, val_loader)

# %%
trainer.test(model, test_loader)

# %%



