import torch 

def eval_model(model, dataloader, loss_func,  metrics, device):
    """ returns test_loss, test_metrics"""
    model.eval()
    metrics.reset()
    test_loss = 0.0
    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_logits = model(x)
            loss = loss_func(y_logits, y)
            test_loss += loss.item()
            y_pred= torch.argmax(y_logits, dim=1)
            metrics.update(y_pred, y)
        return test_loss/len(dataloader), metrics.compute()

    
    
def train_model(model, epochs, train_loader, test_loader, device, metrics, optimizer, loss_func):
    """ returns   results = {
            "train_loss":[],
            "test_loss" : [],
            "train_acc":[],
            "test_acc": []
        }"""
        
    results = {
        "train_loss":[],
        "test_loss" : [],
        "train_acc":[],
        "test_acc": []
    }
    
    best_test_loss = float('inf')
    patience = 5
    epochs_without_improvement = 0
    for epoch in range(epochs):
        model.train()
        metrics.reset()
        train_loss= 0.0
        for x, y in train_loader:
            x,y = x.to(device), y.to(device)
            y_logits = model(x)
            y_pred = torch.argmax(y_logits, dim=1)
            loss = loss_func(y_logits, y)
            metrics.update(y_pred, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)
        train_acc = metrics.compute()
        test_loss, test_acc = eval_model(model=model, dataloader=test_loader, loss_func=loss_func, metrics=metrics, device=device)
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["train_acc"].append(train_acc)
        results["test_acc"].append(test_acc)
        print(
            f'epoch: {epoch+1}/{epochs} '
            f"train loss: {train_loss} "
            f"test loss: {test_loss} "
            f"train acc: {train_acc} "
            f"test acc: {test_acc}"
        )
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("best model saved")
        else:
            epochs_without_improvement +=1
            print(f"no improvement "
                  f"{epochs_without_improvement}/{patience}")
            
            
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break
        
    return results