import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import sys

sys.path.append('.')
from utils.Wound_Data_Loader_RHL import WoundDataset
from models.pointcnn_rhl import get_model


def parse_args():
    parser = argparse.ArgumentParser('PointCNN Wound Classification Training')
    parser.add_argument('--use_cpu', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--num_point', type=int, default=1024, help='number of points')
    parser.add_argument('--k', type=int, default=16, help='number of nearest neighbors')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--log_dir', type=str, default='pointcnn_wounds')
    parser.add_argument('--decay_rate', type=float, default=1e-4)
    
    # Data arguments
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    
    return parser.parse_args()


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    train_bar = tqdm(train_loader, desc=f'Epoch {epoch} Training')
    for points, labels in train_bar:
        # PointCNN expects (B, N, 3)
        points = points.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(points)
        
        # Calculate loss
        loss = criterion(pred, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        pred_choice = pred.argmax(dim=1)
        correct = (pred_choice == labels).sum().item()
        
        total_loss += loss.item() * labels.size(0)
        total_correct += correct
        total_samples += labels.size(0)
        
        # Update progress bar
        train_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct / labels.size(0):.4f}'
        })
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    return avg_loss, avg_acc


def test(model, test_loader, criterion, device, class_names):
    """Test the model"""
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Per-class accuracy
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Testing')
        for points, labels in test_bar:
            points = points.to(device)
            labels = labels.to(device)
            
            # Forward pass
            pred = model(points)
            
            # Calculate loss
            loss = criterion(pred, labels)
            
            # Statistics
            pred_choice = pred.argmax(dim=1)
            correct = (pred_choice == labels).sum().item()
            
            total_loss += loss.item() * labels.size(0)
            total_correct += correct
            total_samples += labels.size(0)
            
            # Per-class statistics
            for i in range(len(labels)):
                label = labels[i].item()
                pred_label = pred_choice[i].item()
                
                if label not in class_total:
                    class_total[label] = 0
                    class_correct[label] = 0
                
                class_total[label] += 1
                if label == pred_label:
                    class_correct[label] += 1
            
            test_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct / labels.size(0):.4f}'
            })
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    # Calculate per-class accuracy
    class_acc = {label: class_correct.get(label, 0) / class_total.get(label, 1) 
                 for label in class_total}
    
    return avg_loss, avg_acc, class_acc


def main(args):
    # Set device
    if args.use_cpu:
        device = torch.device("cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Create log directory
    log_dir = os.path.join('log', 'classification', args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(log_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = WoundDataset(
        json_path=args.json_path,
        data_dir=args.data_dir,
        split='train',
        num_points=args.num_point,
        normalize=True
    )
    
    test_dataset = WoundDataset(
        json_path=args.json_path,
        data_dir=args.data_dir,
        split='test',
        num_points=args.num_point,
        normalize=True
    )
    
    num_classes = train_dataset.num_classes
    class_names = train_dataset.class_names
    
    print(f"\nDataset loaded successfully!")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print(f"\nBuilding PointCNN model...")
    model = get_model(num_classes=num_classes, input_channels=3, k=args.k)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.decay_rate
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.decay_rate
        )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    best_test_acc = 0.0
    best_class_acc = {}
    
    # Save training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for epoch in range(args.epoch):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epoch}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        # Test
        test_loss, test_acc, class_acc = test(model, test_loader, criterion, device, class_names)
        print(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        
        # Print per-class accuracy
        print("\nPer-class accuracy:")
        for label_idx, acc in sorted(class_acc.items()):
            class_name = class_names[label_idx]
            print(f"  {class_name}: {acc:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_class_acc = class_acc
            save_path = os.path.join(log_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"\n✓ Saved best model (acc: {best_test_acc:.4f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(log_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'test_acc': test_acc,
            }, save_path)
            print(f"Saved checkpoint: {save_path}")
        
        # Save training history
        with open(os.path.join(log_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=4)
    
    # Final summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Best test accuracy: {best_test_acc:.4f}")
    print(f"\nBest per-class accuracy:")
    for label_idx, acc in sorted(best_class_acc.items()):
        class_name = class_names[label_idx]
        print(f"  {class_name}: {acc:.4f}")
    print(f"\nModel saved to: {log_dir}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
