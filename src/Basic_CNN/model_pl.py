import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

import torchmetrics

from pytorch_lightning.loggers import TensorBoardLogger

from Basic_CNN import SimpleCNN

class image_classifier(pl.LightningModule):
  def __init__(self, train_set, val_set, batch_size = 8, lr = 1e-3):

    super(image_classifier, self).__init__()

    self.cnn = SimpleCNN()

    self.lr = lr
    
    self.batch_size = batch_size

    self.train_set = train_set
    self.val_set = val_set

    self.loss = nn.NLLLoss()

    self.accuracy = torchmetrics.Accuracy()


  def forward(self, x):
    """compute pass forward"""

    x = self.cnn(x)

    return x

  def __loss(self, y, label):
    """Compute the loss"""
    loss = self.loss(y, label.long())
    
    return loss

  def training_step(self, batch, batch_idx: int):
    """Compute a training step
        Args:
            batch (List[torch.Tensor]): batch 
        Returns:
            OrderedDict: dict {loss, progress_bar}
        """

    x, label = batch
    y = self(x)

    loss = self.__loss(y, label.long())

    self.log('train/train_acc_step', self.accuracy(y,label.long()))
    self.log('train/loss', loss)

    return dict(loss=loss, log=dict(train_loss=loss.detach()))
  

  def training_epoch_end(self, outs) -> None:
    self.log('train/train_acc_epoch', self.accuracy.compute())


  def validation_step(self, batch, batch_idx: int):
    """Compute a validation step
        Args:
            batch (List[torch.Tensor]): batch 
        Returns:
            OrderedDict: dict {loss, progress_bar}
        """

    x, label = batch
    y = self(x)

    loss = self.__loss(y, label.long())

    self.log('val/val_acc_step', self.accuracy(y,label.long()))
    self.log('val/loss', loss)

    return dict(loss=loss, log=dict(train_loss=loss.detach()))


  def val_epoch_end(self, outs) -> None:
    self.log('val/val_acc_epoch', self.accuracy.compute())


  def configure_optimizers(self):
    """Configure both generator and discriminator optimizers
    Returns:
        Tuple(list): (list of optimizers, empty list) 
    """

    opt = torch.optim.Adam(self.cnn.parameters(), lr=self.lr)

    return {
        'optimizer': opt
    }

  def train_dataloader(self):
    return DataLoader(self.train_set,
                        batch_size=self.batch_size,
                        shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_set, batch_size=self.batch_size)