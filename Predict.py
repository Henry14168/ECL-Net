import os
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from torch_geometric.loader import DataLoader

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

try:
    from model import EGNNModel
    from datasets import load_dataset
except ImportError:
    raise ImportError("❌ 请确保 model.py 和 datasets.py 在当前目录下。")

# ==========================================
# 1. 预测配置 (支持多模型集成)
# ==========================================
CONFIG = {
    'model_paths': [
        'results/EGNN/Bio_Expert_Final_20260318_1130/best_ft_seed_1024.pth',
        'results/EGNN/Run_006_S2644_Acc_0.8393.pth',
        'results/EGNN/Run_074_S2644_Acc_0.8611.pth',
        'results/EGNN/Run_089_S2644_Acc_0.8544.pth'
    ],
    'graph_dir': 'data/graphs',
    'split': 'predict',

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 16,

    'input_dim': 2560,
    'hidden_dim': 64,
    'n_layers': 2,
    'dropout_rate': 0.4866,
    'edge_dim': 16
}


def main():
    print(f"🚀 [Info] Using device: {CONFIG['device']}")

    # -------------------------------------------------------
    # 1. 加载预测集
    # -------------------------------------------------------
    print(f"📦 [Info] Loading Dataset: {CONFIG['split']}...")
    try:
        dataset = load_dataset(CONFIG['graph_dir'], CONFIG['split'], labeled=False, include_reverse=False)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    if len(dataset) == 0:
        print("🔴 No data loaded. Check your data/graphs directory.")
        return

    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    mutation_results = defaultdict(dict)
    valid_model_names = []

    # -------------------------------------------------------
    # 2. 遍历所有模型进行预测
    # -------------------------------------------------------
    # 这里为了在字典中存储方便，去掉了路径和后缀，只保留文件名
    for idx, model_path in enumerate(CONFIG['model_paths']):
        model_name = os.path.basename(model_path).replace('.pth', '')

        print(f"\n🛠️  [{idx + 1}/{len(CONFIG['model_paths'])}] Running Model: {model_name}...")

        model = EGNNModel(
            input_dim=CONFIG['input_dim'],
            hidden_dim=CONFIG['hidden_dim'],
            n_layers=CONFIG['n_layers'],
            dropout_rate=CONFIG['dropout_rate'],
            edge_dim=CONFIG['edge_dim']
        ).to(CONFIG['device'])

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=CONFIG['device'])
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']

            model.load_state_dict(state_dict, strict=True)
            valid_model_names.append(model_name)
        else:
            print(f"⚠️  Model path not found, skipping: {model_path}")
            continue

        model.eval()

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(CONFIG['device'])
                outputs = model(batch)

                if isinstance(outputs, (tuple, list)):
                    logits = outputs[0]
                else:
                    logits = outputs

                logits = logits.view(-1)
                probs = torch.sigmoid(logits).cpu().numpy()

                if hasattr(batch, 'name'):
                    names = batch.name
                else:
                    names = [f"Unknown_{i}" for i in range(len(probs))]

                for i in range(len(probs)):
                    name = names[i]
                    p = probs[i]
                    mutation_results[name][model_name] = float(p)

    # -------------------------------------------------------
    # 3. 整理结果、计算综合指标并排序导出
    # -------------------------------------------------------
    if len(mutation_results) > 0 and len(valid_model_names) > 0:
        print("\n" + "=" * 80)
        print(f"📊 AGGREGATING & SORTING PREDICTIONS ({CONFIG['split']})")
        print("=" * 80)

        final_data = []
        for mut, res_dict in mutation_results.items():
            row = {'Mutation': mut}
            probs_list = []

            for m_name in valid_model_names:
                prob = res_dict.get(m_name, np.nan)
                row[m_name] = prob
                if not np.isnan(prob):
                    probs_list.append(prob)

            # ✅ 新增计算逻辑：平均值、标准差、赞同票数
            if probs_list:
                row['Avg_Prob'] = np.mean(probs_list)
                row['Std_Dev'] = np.std(probs_list)
                row['Agree_Count'] = sum(1 for p in probs_list if p > 0.5)
                row['Final_Pred'] = 1 if row['Avg_Prob'] > 0.5 else 0
            else:
                row['Avg_Prob'] = np.nan
                row['Std_Dev'] = np.nan
                row['Agree_Count'] = 0
                row['Final_Pred'] = 0

            final_data.append(row)

        df = pd.DataFrame(final_data)

        # ✅ 核心排序逻辑：按 Avg_Prob 降序排；若相同，按 Std_Dev 升序排（争议越小越靠前）
        df = df.sort_values(by=['Avg_Prob', 'Std_Dev'], ascending=[False, True]).reset_index(drop=True)

        # 调整列的显示顺序，让关键指标靠前
        cols = ['Mutation', 'Avg_Prob', 'Std_Dev', 'Agree_Count', 'Final_Pred'] + valid_model_names
        df = df[cols]

        print("🔝 Top 5 High-Confidence Predictions:")
        # 终端打印时稍微格式化一下浮点数，看着更清爽
        print(df.head(5).to_string(formatters={'Avg_Prob': '{:.4f}'.format, 'Std_Dev': '{:.4f}'.format}))

        save_name = f"ensemble_predictions_robust_{CONFIG['split']}.csv"
        df.to_csv(save_name, index=False)

        print("-" * 80)
        print(f"💾 Robust ensemble predictions successfully saved to: {save_name}")
    else:
        print("🔴 Failed to generate predictions. Please check if model paths are correct.")


if __name__ == "__main__":
    main()