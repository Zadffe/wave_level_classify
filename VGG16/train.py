import argparse
import logging
import torch
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from models.vgg import vgg16_pretrained, vgg16_bn
from utils.dataset import get_data_loaders
from config import default_config
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(filename=log_path,
                        format='%(asctime)s %(message)s',
                        filemode='w',
                        level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)


# 测试代码
def evaluate(model, test_loader, device, criterion, class_names):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(test_loader)
    acc = accuracy_score(all_labels, all_preds)
    report_dict = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names, 
        output_dict=True,
        digits=4
        )
    return avg_loss, acc, report_dict, all_preds, all_labels



def plot_confusion_matrix(y_true, y_pred, class_names, filename, save_dir):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 绘制混淆矩阵
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix (%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    
    # 记录具体数值
    logging.info("\nConfusion Matrix (count):")
    logging.info(f"\n{cm}")
    logging.info("\nConfusion Matrix (percentage):")
    logging.info(f"\n{cm_percent}")



def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 释放cuda缓存
    torch.cuda.empty_cache()


    setup_logger(args.log_path)
    logging.info("=== Loading data ===")
    train_loader, test_loader, class_names = get_data_loaders(args.data_dir, args.batch_size,args.num_workers)

    model = vgg16_bn(num_classes=len(class_names)).to(device)

    # 打印模型结构
    # logging.info("=== Model Summary ===")

    # logging.info("Model Structure:")
    # logging.info("\n" + str(model))
    # # summary(model, input_size=(3, 224, 224))
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"\nTotal parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_losses, test_accs, test_losses = [], [], []
    os.makedirs("results", exist_ok=True)

    total_training_time= 0


    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )
    
    # 早停相关变量
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_confusion_matrix_data = None
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0
        start_time = time.time()

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch} /{args.epochs} Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # 计算当前轮训练时间
        epoch_time = time.time() - start_time
        # 总时间
        total_training_time += epoch_time


        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logging.info(f"Epoch time: {epoch_time / 60:.2f}m {epoch_time % 60:.2f}s")


        # 每轮结束后在测试集上评估
        avg_test_loss, acc, report_dir, all_preds, all_labels = evaluate(model, test_loader, device, criterion, class_names)
        test_losses.append(avg_test_loss)
        test_accs.append(acc)
        logging.info(
        f"Epoch [{epoch}/{args.epochs}]  "
        f"Train Loss: {avg_train_loss:.4f}  "
        f"Test Loss: {avg_test_loss:.4f}  "
        f"Test Acc: {acc:.4f}  "
        )

        for cls in class_names:
            cls_metrics = report_dir[cls]
            logging.info(f"{cls} "
                         f"Precision: {cls_metrics['precision']:.4f}  "
                         f"Recall: {cls_metrics['recall']:.4f} " 
                         f"F1-score: {cls_metrics['f1-score']:.4f}"
                         f"(Support: {int(cls_metrics['support'])})")
        
        # 在每个epoch结束后更新学习率
        scheduler.step(avg_test_loss)
        
        # 早停检查
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_confusion_matrix_data = (all_labels, all_preds)
            best_epoch = epoch
            # 保存最佳模型
            torch.save(model.state_dict(), args.model_path)
            logging.info(f"Best model saved at epoch {epoch} ")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch}")
                break
    
    # # 保存模型
    # torch.save(model.state_dict(), args.model_path)
    # logging.info(f"Model saved to {args.model_path}")

    os.makedirs(args.result_dir, exist_ok=True)


    # 绘制除损失之外的曲线
    def plot_para_curve(values, title, ylabel, filename):
        plt.figure()
        plt.plot(range(1, len(values)+1), values, marker="o")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(os.path.join(args.result_dir, filename))
        plt.close()
    def plot_loss_curve(train_losses, test_losses, title, filename):
        plt.figure()
        epochs = range(1, len(train_losses)+1)
        plt.plot(epochs, train_losses, marker="o", label="Train Loss")
        plt.plot(epochs, test_losses, marker="s", label="Test Loss")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.result_dir, filename))
        plt.close()
    plot_loss_curve(train_losses, test_losses, "Train Loss vs Test Loss", "loss.png")
    plot_para_curve(test_accs,    "Test Accuracy",     "Accuracy", "test_accuracy.png")
    
    # 使用最佳模型的数据绘制混淆矩阵
    if best_confusion_matrix_data:
        best_labels, best_preds = best_confusion_matrix_data
        plot_confusion_matrix(
            best_labels, 
            best_preds, 
            class_names, 
            f"confusion_matrix_best_epoch_{best_epoch}.png",
            args.result_dir
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wave Classification Training")
    parser.add_argument("--data_dir", type=str, default=default_config["data_dir"])
    parser.add_argument("--batch_size", type=int, default=default_config["batch_size"])
    parser.add_argument("--epochs", type=int, default=default_config["epochs"])
    parser.add_argument("--learning_rate", type=float, default=default_config["learning_rate"])
    parser.add_argument("--log_path", type=str, default=default_config["log_path"])
    parser.add_argument("--model_path", type=str, default=default_config["model_path"])
    parser.add_argument("--num_workers", type=int, default=default_config["num_workers"])
    parser.add_argument("--result_dir", type=str, default=default_config["result_dir"])
    args = parser.parse_args()

    train(args)
