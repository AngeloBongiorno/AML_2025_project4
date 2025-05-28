import torch
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchmetrics.segmentation import MeanIoU
import os
import zipfile


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

          valid_mask = (y_val >= 0) & (y_val < 7)

          preds_flat = preds[valid_mask]
          targets_flat = y_val[valid_mask]

          # Aggiorna il valore del mIoU
          miou.update(preds_flat, targets_flat)

          loss = loss_fn(outputs, y_val)

          val_loss += loss.item()

  avg_val_loss = val_loss / len(val_loader)
  miou_score = miou.compute().mean()    # Ottieni il valore scalare di mIoU

  return avg_val_loss, miou_score



plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])



def get_loveDA(test_set=False, verbose=False, train_cut=False):

  """Downloads the loveDA dataset. If test_set == True, also downloads the test set.
  If verbose == True activates verbose mode.
  If train_cut == True, also returns the stylized version of the urban training set (urban -> rural).
  Returns a dictionary with the paths to the downloaded data."""

  
  training_set_path = "/content/drive/My Drive/AML_project/Train.zip"

  # ZIP files paths on Google Drive  
  if test_set == True:
    zip_files = {
    "training": training_set_path,
    "validation": "/content/drive/My Drive/AML_project/Val.zip",
    "test": "/content/drive/My Drive/AML_project/Test.zip"
  }
  else:
    zip_files = {
        "training": training_set_path,
        "validation": "/content/drive/My Drive/AML_project/Val.zip",
    }

  
  if train_cut:
    training_set_cut_path = "/content/drive/My Drive/AML_project/Train_CUT.zip"
    zip_files["training_cut"] = training_set_cut_path

  # Destination directory on Colab
  extract_path = "/content/dataset"

  # Create the directory if it doesn't exist
  os.makedirs(extract_path, exist_ok=True)


  extract_dir = f"{extract_path}"

  
  # Check if the directory is non-empty (assumes extraction is complete if the folder has files)
  if os.path.exists(extract_dir) and any(os.scandir(extract_dir)):
    print(f"Skipping extraction for {name}, already extracted.")
  else:
    for name, zip_path in zip_files.items():
        if verbose:
          print(f"Extracting {name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        if verbose:
          print(f"{name} extracted!")

    if verbose:
      print("Extraction check completed!")
      

  TRAINING_PATH_URBAN = os.path.join(extract_path, "Train", "Urban")
  TRAINING_PATH_RURAL = os.path.join(extract_path, "Train", "Rural")
  VAL_PATH_URBAN = os.path.join(extract_path, "Val", "Urban")
  VAL_PATH_RURAL = os.path.join(extract_path, "Val", "Rural")

  
  

  

  paths_dict = {
    "training_urban": TRAINING_PATH_URBAN,
    "training_rural": TRAINING_PATH_RURAL,
    "validation_urban": VAL_PATH_URBAN,
    "validation_rural": VAL_PATH_RURAL,
  }


  if train_cut:
    TRAINING_PATH_URBAN_CUT = os.path.join(extract_path, "Train_CUT", "Urban")
    paths_dict["training_urban_cut"] = TRAINING_PATH_URBAN_CUT
  

  if test_set == True:
    TEST_PATH_URBAN = os.path.join(extract_path, "Test", "Urban")
    TEST_PATH_RURAL = os.path.join(extract_path, "Test", "Rural")
    paths_dict["test_urban"] = TEST_PATH_URBAN
    paths_dict["test_rural"] = TEST_PATH_RURAL

  return paths_dict
