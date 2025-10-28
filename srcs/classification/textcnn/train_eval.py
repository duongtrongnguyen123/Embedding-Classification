import torch

def acc_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
                epochs=12, patience=3, save_path="best_textcnn.pt"):
    best_val, wait = 0.0, 0

    # 🔹 Mở file log với encoding UTF-8
    log_file = open("train_log.txt", "w", encoding="utf-8")
    log_file.write("Epoch | Train_Acc | Val_Acc | Train_Loss | Val_Loss\n")
    log_file.write("------------------------------------------------------\n")

    for epoch in range(epochs):
        model.train()
        train_loss = train_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            train_acc += acc_from_logits(logits, y_batch) * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)

        model.eval()
        val_loss = val_acc = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_acc += acc_from_logits(logits, y_batch) * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        scheduler.step()

        # 🔹 Ghi log mỗi epoch ra file
        log_line = f"Epoch {epoch+1:02d}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}\n"
        log_file.write(log_line)
        print(log_line.strip())

        # 🔹 Lưu mô hình tốt nhất
        if val_acc > best_val:
            best_val, wait = val_acc, 0
            torch.save(model.state_dict(), save_path)
            log_file.write(f"  -> Saved new best model: {save_path}\n")  # dùng "->" thay "→"
            print(f"  -> Saved new best model: {save_path}")
        else:
            wait += 1
            if wait >= patience:
                log_file.write("Early stopping!\n")
                print("Early stopping!")
                break

    log_file.write(f"\nBest Validation Accuracy: {best_val:.4f}\n")
    log_file.close()
    print("✅ Training log saved to train_log.txt")
