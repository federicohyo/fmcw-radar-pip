import torch

from tqdm import tqdm

from configurations_grid import N_EPOCHS, criterion

from sklearn.metrics import confusion_matrix

import numpy as np

def train_model(model, dataloader, optimizer, args, file_stats, train_logs, epoch, qat = False):

    model.train()
    train_loss = 0.0
    correct = 0
    num_example = 0

    # For confusion matrix
    true_labels = []
    pred_labels = []

    device = 'cpu'
    if qat == False:
        device = 'cuda'
        model.to(device)
    if qat == True:
        device = 'cpu'
        model.to(device)

    # Used optimzier by L.Muller et al
    # optimizer = torch.optim.Adam(model.parameters(), lr = 10e-2, weight_decay = 1e-5 )

    # But this was a bit to high.
    # optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)

    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005, weight_decay = 1e-5)
    # steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.3)

    list_chunk_idx = []

    # timing for subcomponents of batch
    for batch in tqdm(dataloader, desc = f"Epoch {epoch + 1} in train", leave = False, disable = args.not_show_progress):
        x, y, label, file, chunk_idx, _, _ = batch
        
        # when y and label are not equal throw error
        if not torch.equal(y, label):
            print(y, label)
            raise ValueError("y and label are not equal")

        list_chunk_idx.extend(np.array(chunk_idx).tolist())

        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)

        # Implementation following L.Muller et al
        # loss = torch.nn.functional.cross_entropy(y_hat, y)

        loss = torch.nn.CrossEntropyLoss()(y_hat, y)

        train_loss += loss.detach().cpu().item() / len(dataloader)
        
        pred = torch.argmax(y_hat, dim = 1)
        correct += (pred == y).sum()
        num_example += len(y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.debugging_mode_2:
            break

    accuracy = correct / num_example
    train_logs['acc_train'].append(accuracy.item())
    train_logs['loss_train'].append(train_loss)
    print(f" Train: Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.3f} acc: {accuracy:.3f}")
    file_stats.write(f"Train: Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.3f} acc: {accuracy:.3f}\n")

    unique_elements_set = set(list_chunk_idx)
    total_unique_elements = len(unique_elements_set)

    # print("Total unique elements:", total_unique_elements)


def validate_model(model, dataloader, args, file_stats, train_logs, epoch, date_time = None, set = 'valid'):
    
    model.eval()
    correct = 0
    num_example = 0
    valid_loss = 0

    true_labels = []
    pred_labels = []
    dataloader_idx = []

    device = next(model.parameters()).device

    for batch in tqdm(dataloader, desc = f"Epoch {epoch + 1} in valid", leave = False, disable = args.not_show_progress):
        x, y, label, _, _, idx, _ = batch
        # when y and label are not equal throw error
        if not torch.equal(y, label):
            print(y, label)
            raise ValueError("y and label are not equal")

        x = x.to(device = device)
        y = y.to(device = device)

        y_hat = model(x)

        loss = criterion(y_hat, y.long())
        valid_loss += loss.detach().cpu().item() / len(dataloader)
        
        pred = torch.argmax(y_hat, dim=1)
        
        correct += (pred == y).sum()
        num_example += len(y)

        true_labels.extend(y.cpu().numpy())
        pred_labels.extend(pred.cpu().numpy())
        dataloader_idx.extend(idx.cpu().numpy())

        if args.debugging_mode_2:
            break

    accuracy = correct / num_example
    train_logs[f'acc_{set}'].append(accuracy.item())
    train_logs[f'loss_{set}'].append(valid_loss)
    print(f" {set}: Epoch {epoch + 1}/{N_EPOCHS} loss: {valid_loss:.3f} acc: {accuracy:.3f}")
    file_stats.write(f"{set}: Epoch {epoch + 1}/{N_EPOCHS} loss: {valid_loss:.3f} acc: {accuracy:.3f}\n")
    is_best = accuracy > train_logs[f'best_acc_{set}']
    train_logs[f'best_acc_{set}'] = max(accuracy, train_logs[f'best_acc_{set}'])

    cm = confusion_matrix(true_labels, pred_labels)

    if is_best:
        train_logs[f'best_confusion_matrix_{set}'] = cm
        print(f'best_accuracy_{set}', train_logs[f'best_acc_{set}'])
        file_stats.write(f"best_accuracy_{set}: {train_logs[f'best_acc_{set}']}\n")
        train_logs[f'file_name_best_model_{set}'] = f'{date_time}_acc_{accuracy:.4f}model_{set}.pt'
        # file_name_best_model = f'{date_time}_acc_{accuracy:.4f}model.pt'
        print(f"Saving best model {set}:", train_logs[f'file_name_best_model_{set}'])

        print(f'./models/{date_time}_acc_{accuracy}model_{set}.pt')
        # torch.save(model, f'./models/{date_time}_acc_{accuracy:.4f}model.pt')
        torch.save(model.state_dict(), f'./models/{date_time}_acc_{accuracy:.4f}model_{set}.pt')

        train_logs['best_valid_model_predictions']['true_labels'] = true_labels
        train_logs['best_valid_model_predictions']['pred_labels'] = pred_labels
        train_logs['best_valid_model_predictions']['loader_idx'] = dataloader_idx

    return accuracy, train_logs

def test_model():
    pass

# def test_model(model, dataloader, args, file_stats, train_logs, epoch, date_time = None):
    
#     model.eval()
#     correct = 0
#     num_example = 0
#     valid_loss = 0

#     true_labels = []
#     pred_labels = []

#     device = next(model.parameters()).device

#     for batch in tqdm(dataloader, desc = f"Epoch {epoch + 1} in test", leave = False, disable = args.not_show_progress):
#         x, y, label, _, _, _, _ = batch
#         # when y and label are not equal throw error
#         if not torch.equal(y, label):
#             print(y, label)
#             raise ValueError("y and label are not equal")

#         x = x.to(device = device)
#         y = y.to(device = device)

#         y_hat = model(x)

#         loss = criterion(y_hat, y.long())
#         valid_loss += loss.detach().cpu().item() / len(dataloader)
        
#         pred = torch.argmax(y_hat, dim=1)
        
#         correct += (pred == y).sum()
#         num_example += len(y)

#         true_labels.extend(y.cpu().numpy())
#         pred_labels.extend(pred.cpu().numpy())

#         if args.debugging_mode_2:
#             break

#     accuracy = correct / num_example
#     train_logs['acc_test'].append(accuracy.item())
#     train_logs['loss_test'].append(valid_loss)
#     print(f" Test: Epoch {epoch + 1}/{N_EPOCHS} loss: {valid_loss:.3f} acc: {accuracy:.3f}")
#     file_stats.write(f"Test: Epoch {epoch + 1}/{N_EPOCHS} loss: {valid_loss:.3f} acc: {accuracy:.3f}\n")

#     cm = confusion_matrix(true_labels, pred_labels)



#     return accuracy, train_logs 


def calibrate_model_ptq(model, dataloader, args):

    print("Calibrating model for PTQ")
    
    model.eval()
    device = next(model.parameters()).device

    for batch in tqdm(dataloader, desc = "Calibrating model for PTQ", leave = False):
        x, y, label, _, _, _, _ = batch
        # when y and label are not equal throw error
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)

        if args.debugging_mode_2:
            break
    
    return model
