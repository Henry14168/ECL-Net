import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from datetime import datetime
from Datasets import load_dataset
from Model import EGNNModel

CONFIG = {
    'graph_dir': 'data/graphs',
    'save_dir': f'results/EGNN',
    'device': 'cuda',

    'base_seed': 1024,
    'num_runs': 1,

    'input_dim': 2560,
    'hidden_dim': 64,
    'n_layers': 2,
    'dropout_rate': 0.4866,
    'edge_dim': 16,

    'pre_lr': 1e-05,
    'con_weight': 0.5,
    'temperature': 0.1,
    'pre_batch_size': 32,
    'pre_epochs': 50,

    'ft_epochs': 10,
    'ft_batch_size': 16,
    'ft_lr': 1e-05,
    'weight_decay': 2e-05,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def calculate_metrics(logits, targets):
    if len(np.unique(targets)) < 2:
        return {'accuracy': 0.0, 'auc': 0.5, 'f1': 0, 'precision': 0, 'recall': 0}
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    auc = roc_auc_score(targets, targets)
    preds = (probs > 0.5).astype(int)
    targets = np.array(targets).astype(int)
    return {
        'accuracy': accuracy_score(targets, preds),
        'auc': auc,
        'f1': f1_score(targets, preds, zero_division=0),
        'precision': precision_score(targets, preds, zero_division=0),
        'recall': recall_score(targets, preds, zero_division=0)
    }


def contrastive_loss(feat1, feat2, temp):
    feat1, feat2 = F.normalize(feat1, dim=1), F.normalize(feat2, dim=1)
    logits = torch.matmul(feat1, feat2.T) / temp
    return F.cross_entropy(logits, torch.arange(feat1.size(0), device=feat1.device))


def train_epoch(model, loader, optimizer):
    model.train()
    for batch in loader:
        batch = batch.to(CONFIG['device'])
        optimizer.zero_grad()
        logits, wt_pair, mut_pair = model(batch)

        loss = F.binary_cross_entropy_with_logits(logits.view(-1), batch.y.float().view(-1))

        if wt_pair[0].size(0) > 1:
            loss += CONFIG['con_weight'] * (contrastive_loss(wt_pair[0], wt_pair[1], CONFIG['temperature']) +
                                            contrastive_loss(mut_pair[0], mut_pair[1], CONFIG['temperature']))
        loss.backward()
        optimizer.step()


def evaluate(model, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            logits = out[0] if isinstance(out, tuple) else out
            all_logits.extend(logits.view(-1).cpu().numpy())
            all_targets.extend(batch.y.view(-1).cpu().numpy())
    return calculate_metrics(all_logits, all_targets)


def run_single_experiment(run_idx, data_S2648, data_S211, data_TG):
    seed = random.randint(0, 100000)
    set_seed(seed)
    device = CONFIG['device']

    num_pairs_pre = len(data_S2648) // 2
    idx_temp_pairs, idx_test_pairs = train_test_split(np.arange(num_pairs_pre), test_size=0.2,
                                                      random_state=seed)
    idx_train_pairs, idx_val_pairs = train_test_split(idx_temp_pairs, test_size=0.125, random_state=seed)

    idx_pre_train = [j for i in idx_train_pairs for j in (2 * i, 2 * i + 1)]
    idx_pre_val = [j for i in idx_val_pairs for j in (2 * i, 2 * i + 1)]
    idx_pre_test = [j for i in idx_test_pairs for j in (2 * i, 2 * i + 1)]

    loader_pre_train = DataLoader([data_S2648[i] for i in idx_pre_train], batch_size=CONFIG['pre_batch_size'],
                                  shuffle=True)
    loader_pre_val = DataLoader([data_S2648[i] for i in idx_pre_val], batch_size=CONFIG['pre_batch_size'],
                                shuffle=False)
    loader_pre_test = DataLoader([data_S2648[i] for i in idx_pre_test], batch_size=CONFIG['pre_batch_size'],
                                 shuffle=False)

    num_pairs_ft = len(data_S211) // 2
    idx_ft_train_pairs, idx_ft_val_pairs = train_test_split(np.arange(num_pairs_ft), test_size=0.1, random_state=seed)

    idx_ft_train = [j for i in idx_ft_train_pairs for j in (2 * i, 2 * i + 1)]
    idx_ft_val = [j for i in idx_ft_val_pairs for j in (2 * i, 2 * i + 1)]

    loader_ft_train = DataLoader([data_S211[i] for i in idx_ft_train], batch_size=CONFIG['ft_batch_size'], shuffle=True)
    loader_ft_val = DataLoader([data_S211[i] for i in idx_ft_val], batch_size=CONFIG['ft_batch_size'], shuffle=False)

    loader_TG = DataLoader(data_TG, batch_size=CONFIG['ft_batch_size'], shuffle=False)

    model = EGNNModel(input_dim=CONFIG['input_dim'], hidden_dim=CONFIG['hidden_dim'],
                      n_layers=CONFIG['n_layers'], dropout_rate=CONFIG['dropout_rate'], edge_dim=CONFIG['edge_dim']).to(
        device)

    optimizer_pre = optim.Adam(model.parameters(), lr=CONFIG['pre_lr'])
    best_pre_acc = 0.0
    temp_pre_path = os.path.join(CONFIG['save_dir'], f"temp_pre_{run_idx}.pth")
    pre_patience = 0

    for ep in range(CONFIG['pre_epochs']):
        train_epoch(model, loader_pre_train, optimizer_pre)
        val_m = evaluate(model, loader_pre_val, device)
        if val_m['accuracy'] > best_pre_acc:
            best_pre_acc = val_m['accuracy']
            torch.save(model.state_dict(), temp_pre_path)
            pre_patience = 0
        else:
            pre_patience += 1

        if pre_patience >= 6:
            break

    model.load_state_dict(torch.load(temp_pre_path))
    metrics_s2648 = evaluate(model, loader_pre_test, device)

    optimizer_ft = optim.Adam(model.parameters(), lr=CONFIG['ft_lr'], weight_decay=CONFIG['weight_decay'])
    best_ft_val_acc = 0.0
    save_name = f"Run_{run_idx:03d}_S2648_Acc_{metrics_s2648['accuracy']:.2f}.pth"
    seed_ft_path = os.path.join(CONFIG['save_dir'], save_name)
    ft_patience = 0

    for ep in range(CONFIG['ft_epochs']):
        train_epoch(model, loader_ft_train, optimizer_ft)
        val_ft_m = evaluate(model, loader_ft_val, device)
        if val_ft_m['accuracy'] > best_ft_val_acc:
            best_ft_val_acc = val_ft_m['accuracy']
            torch.save(model.state_dict(), seed_ft_path)
            ft_patience = 0
        else:
            ft_patience += 1

        if ft_patience >= 6:
            break

    model.load_state_dict(torch.load(seed_ft_path))
    metrics_tg = evaluate(model, loader_TG, device)

    if os.path.exists(temp_pre_path):
        os.remove(temp_pre_path)

    return metrics_s2648, metrics_tg


def main():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    log_file_path = os.path.join(CONFIG['save_dir'], "experiment_results_log.txt")

    data_S2648 = load_dataset(CONFIG['graph_dir'], 'S2648', labeled=True, include_reverse=True)
    data_S211 = load_dataset(CONFIG['graph_dir'], 'S211', labeled=True, include_reverse=True)
    data_TG = load_dataset(CONFIG['graph_dir'], 'TG', labeled=True, include_reverse=False)

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment Start Time: {datetime.now()}\n")
        f.write("Run_ID\tS2648_Acc\tS2648_Pre\tS2648_Rec\tS2648_F1\tS2648_AUC\tTG_Acc\tTG_Pre\tTG_Rec\tTG_F1\tTG_AUC\n")

    print(f" Starting {CONFIG['num_runs']} runs. Results will be logged to {log_file_path}")

    for i in range(CONFIG['num_runs']):
        run_idx = i + 1
        s_m, t_m = run_single_experiment(run_idx, data_S2648, data_S211, data_TG)

        log_line = (
            f"{run_idx}\t{s_m['accuracy']:.2f}\t{s_m['precision']:.3f}\t{s_m['recall']:.3f}\t{s_m['f1']:.3f}\t{s_m['auc']:.3f}\t"
            f"{t_m['accuracy']:.3f}\t{t_m['precision']:.3f}\t{t_m['recall']:.3f}\t{t_m['f1']:.3f}\t{t_m['auc']:.3f}\n")

        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(log_line)

        print(f" Run {run_idx}/{CONFIG['num_runs']} Finished. S2648_AUC: {s_m['auc']:.3f} | TG_AUC: {t_m['auc']:.3f}")

    print(f"\n All experiments finished. Check your folder: {CONFIG['save_dir']}")


if __name__ == "__main__":
    main()