import torch
import torchmetrics

def train_step(model: torch.nn.Module,
               optimizer: torch.optim,
               loss_fn: torch.nn.Module,
               train_loader: torch.utils.data.DataLoader,
               device: torch.device,
               epoch: int,
               EPOCHS: int,
               ):
  
  """ Trains the network for a single step, returns the average training loss for the step"""
  
  model.to(device)
  model.train()
  train_loss = 0

  # Barra di avanzamento per i batch con informazioni sull'epoca
  #batch_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=100)

  for batch, (X, y) in enumerate(train_loader):
      X = X.to(device)
      y = y.to(device)

      outputs = model(X)

      # WARNING: indicizzare [0] funziona con deeplabv2 in training mode,
      # potrebbe dover essere eliminato per modelli diversi
      loss = loss_fn(outputs[0], y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      train_loss += loss.item()

      # Aggiorna la barra di avanzamento con la loss corrente
      #batch_pbar.set_postfix(loss=loss.item())

  avg_train_loss = train_loss / len(train_loader)

  return avg_train_loss



def validation_step(model: torch.nn.Module,
               loss_fn: torch.nn.Module,
               val_loader: torch.utils.data.DataLoader,
               device: torch.device,
               miou: torchmetrics.segmentation.MeanIoU):

  """Validates the network for a single step, returns the average validation loss for the step and the mIoU score"""

  model.to(device)
  model.eval()
  val_loss = 0
  miou.reset()

  with torch.inference_mode():
      for X_val, y_val in val_loader:
          X_val = X_val.to(device)
          y_val = y_val.to(device)

          outputs = model(X_val)

          # Converte le predizioni nei valori di classe più probabili
          preds = torch.argmax(outputs, dim=1)

          # Aggiorna il valore del mIoU
          miou.update(preds, y_val)

          loss = loss_fn(outputs, y_val)

          val_loss += loss.item()

  avg_val_loss = val_loss / len(val_loader)
  miou_score = miou.compute().item()    # Ottieni il valore scalare di mIoU

  return avg_val_loss, miou_score



