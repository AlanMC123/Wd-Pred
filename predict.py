"""
单词难度预测模型的预测可视化工具。
包含：ROC 曲线绘制、散点图绘制。
新增核心功能：【失败样本深度诊断】(Failure Case Analysis)，用于分析模型为何不敢预测"输"。
修复记录：
- 修复 Transformer 加载时因缺少 focal_loss 导致的报错。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import random
import os 
import pandas as pd
import tensorflow as tf

# 设置随机种子以确保可复现性
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

set_seed(SEED)

def calculate_auc_best(y_true, prob):
    """
    计算 prob 和 -prob 的 AUC；返回 (best_auc, used_prob, inverted_flag)
    """
    try:
        auc_pos = roc_auc_score(y_true, prob)
    except Exception:
        auc_pos = float("nan")
    try:
        auc_neg = roc_auc_score(y_true, -prob)
    except Exception:
        auc_neg = float("nan")

    if np.isnan(auc_pos) and np.isnan(auc_neg):
        return float("nan"), prob, False
    if np.isnan(auc_pos):
        return auc_neg, -prob, True
    if np.isnan(auc_neg):
        return auc_pos, prob, False

    if auc_neg > auc_pos:
        return auc_neg, -prob, True
    else:
        return auc_pos, prob, False

def analyze_failure_cases(y_true_steps, pred_steps, pred_prob, model_name="Model"):
    """
    【核心诊断函数】
    深入分析实际失败（步数>6）的样本中，模型表现如何。
    计算漏报率，并统计错误样本的预测概率分布，帮助确定最佳阈值。
    """
    y_true = y_true_steps.flatten()
    p_steps = pred_steps.flatten()
    p_prob = pred_prob.flatten()
    
    # 1. 找出所有实际失败的样本 (步数 > 6.0)
    fail_mask = (y_true > 6.0)
    total_failures = np.sum(fail_mask)
    
    print(f"\n{'='*20} [{model_name}] Failure Analysis {'='*20}")
    
    if total_failures == 0:
        print("No actual failure samples (steps > 6) found in this dataset.")
        return

    # 提取这些失败样本的预测值
    probs_of_failures = p_prob[fail_mask]
    steps_of_failures = p_steps[fail_mask]

    # 2. 计算漏报率：原本输了，模型却预测赢 (Prob > 0.5)
    default_threshold = 0.5
    missed_mask = (probs_of_failures > default_threshold)
    num_missed = np.sum(missed_mask)
    
    missed_probs = probs_of_failures[missed_mask]
    missed_steps = steps_of_failures[missed_mask]
    
    print(f"📉 Total Actual Failures: {total_failures}")
    print(f"❌ False Positives (Predicted Win but Actually Lost): {num_missed}")
    print(f"⚠️  Failure Misclassification Rate: {num_missed/total_failures:.2%} (Threshold={default_threshold})")
    
    if len(missed_probs) > 0:
        print(f"\n🔍 Diagnosis of the {len(missed_probs)} misclassified samples:")
        print(f"   [Probability Stats] (Model confidence in wrong prediction)")
        print(f"   - Mean:   {np.mean(missed_probs):.4f}")
        print(f"   - Median: {np.median(missed_probs):.4f}")
        print(f"   - Min:    {np.min(missed_probs):.4f}")
        print(f"   - Max:    {np.max(missed_probs):.4f}")
        
        print(f"   [Step Regression Stats] (Does the regression head know better?)")
        print(f"   - Mean:   {np.mean(missed_steps):.4f}")
        print(f"   - Max:    {np.max(missed_steps):.4f}")
        
        # 3. 模拟阈值调整效果
        print(f"\n🛠️  Simulation: Adjusting Decision Logic")
        
        # 方案 A: 提高概率阈值
        for t in [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
            caught = np.sum(probs_of_failures <= t)
            print(f"   - If Threshold = {t:.2f}: Caught {caught}/{total_failures} failures ({caught/total_failures:.1%})")
                
        # 方案 B: 混合逻辑
        hybrid_caught = 0
        for p, s in zip(probs_of_failures, steps_of_failures):
            if p < 0.85 and s > 5.5:
                hybrid_caught += 1
            elif p <= 0.5:
                hybrid_caught += 1
        
        print(f"   - Hybrid Logic (Prob < 0.85 & Steps > 5.5): Caught {hybrid_caught}/{total_failures} failures ({hybrid_caught/total_failures:.1%})")
    
    print("="*60)

def plot_roc_curve(y_true, prob, save_path, model_name="Model"):
    auc, used_prob, inverted = calculate_auc_best(y_true, prob)
    if inverted:
        print(f"⚠️ ROC plotting: detected better AUC with -prob for {model_name}, using -prob for ROC plot.")
    
    fpr, tpr, _ = roc_curve(y_true, used_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {save_path}")

def add_jitter(data, jitter_amount=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    jitter = np.random.normal(0, jitter_amount, size=data.shape)
    return data + jitter

def plot_scatter(y_true, y_pred, save_path, model_name="Model", jitter_amount=0.18):
    plt.figure(figsize=(10, 8))
    errors = np.abs(y_true - y_pred)
    y_true_jitter = add_jitter(y_true, jitter_amount=jitter_amount, seed=SEED)
    y_pred_jitter = add_jitter(y_pred, jitter_amount=jitter_amount, seed=SEED+1)
    
    plt.scatter(y_true_jitter, y_pred_jitter, c=errors, cmap='viridis', alpha=0.3, s=20, label='Prediction points')
    cbar = plt.colorbar()
    cbar.set_label('Prediction Error (|True - Predicted|)')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Ideal prediction line (y=x)')
    plt.title(f'{model_name} Model: Predicted vs True Values')
    plt.xlabel('True Values (Steps)')
    plt.ylabel('Predicted Values (Steps)')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved to: {save_path}")

def safe_read_csv(path, usecols=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File missing: {path}")
    return pd.read_csv(path, usecols=usecols)

# ==========================================================
# 主程序
# ==========================================================
if __name__ == "__main__":
    print("Starting Model Evaluation and Diagnostics...")
    
    # -------------------
    # 1. LSTM Model Evaluation
    # -------------------
    try:
        print("\n=== Processing LSTM Model ===")
        from train_LSTM import (
            load_tokenizer as load_lstm_tok, attach_features as attach_lstm_feat,
            build_history as build_lstm_hist, focal_loss,
            MODEL_SAVE_PATH as LSTM_PATH, LOOK_BACK as LSTM_LB,
            MAX_TRIES, GRID_FEAT_LEN
        )
        
        lstm_model = tf.keras.models.load_model(LSTM_PATH, custom_objects={'focal_loss(gamma=2.0,alpha=0.25)': focal_loss(alpha=0.25, gamma=2.0)})
        lstm_tok = load_lstm_tok()
        
        train_df = safe_read_csv("dataset/train_data.csv", ["Game", "Trial", "Username", "target", "processed_text"])
        val_df = safe_read_csv("dataset/val_data.csv", ["Game", "Trial", "Username", "target", "processed_text"])
        test_df = safe_read_csv("dataset/test_data.csv", ["Game", "Trial", "Username", "target", "processed_text"])
        
        u_df = safe_read_csv("dataset/player_data.csv")
        d_df = safe_read_csv("dataset/difficulty.csv")
        u_map = dict(zip(u_df["Username"], u_df["avg_trial"]))
        d_map = dict(zip(d_df["word"], d_df["avg_trial"]))
        
        val_df = attach_lstm_feat(val_df, lstm_tok, u_map, d_map)
        test_df = attach_lstm_feat(test_df, lstm_tok, u_map, d_map)
        val_hist = build_lstm_hist(val_df)
        test_hist = build_lstm_hist(test_df)
        
        def create_lstm_samples_local(history, look_back):
            X_seq, X_wid, X_bias, X_diff, X_grid, y_st, y_su = [], [], [], [], [], [], []
            for u, evs in history.items():
                if len(evs) <= look_back: continue
                for i in range(look_back, len(evs)):
                    win = evs[i-look_back:i]
                    tgt = evs[i]
                    tr = np.array([t[0] for t in win], np.float32)
                    norm, std = tr/7.0, np.std(tr)/7.0
                    X_seq.append(np.stack([norm, np.full_like(norm, std)], axis=1))
                    X_wid.append([tgt[1]])
                    X_bias.append([tgt[2]/7.0])
                    X_diff.append([tgt[3]/7.0])
                    X_grid.append(tgt[4])
                    y_st.append(min(float(tgt[0]), 7.0))
                    y_su.append(1.0 if tgt[0] <= 6 else 0.0)
            if not X_seq: return None
            return (np.array(X_seq, np.float32), np.array(X_wid, np.int32), np.array(X_bias, np.float32),
                    np.array(X_diff, np.float32), np.array(X_grid, np.float32), np.array(y_st, np.float32), np.array(y_su, np.float32))

        val_data = create_lstm_samples_local(val_hist, LSTM_LB)
        test_data = create_lstm_samples_local(test_hist, LSTM_LB)
        
        if val_data:
            print("LSTM: Predicting Validation Set...")
            v_p_steps, v_p_prob = lstm_model.predict({
                "input_history": val_data[0], "input_word_id": val_data[1],
                "input_user_bias": val_data[2], "input_difficulty": val_data[3],
                "input_grid_sequence": val_data[4]
            }, batch_size=1024, verbose=1)
            analyze_failure_cases(val_data[5], v_p_steps, v_p_prob, model_name="LSTM (Val)")
            plot_roc_curve(val_data[6], v_p_prob.flatten(), "visualization/LSTM_validation_roc_curve.png", "LSTM")
            plot_scatter(val_data[5], np.clip(v_p_steps.flatten(), 0, 7), "visualization/LSTM_validation_scatter.png", "LSTM")

        if test_data:
            print("LSTM: Predicting Test Set...")
            t_p_steps, t_p_prob = lstm_model.predict({
                "input_history": test_data[0], "input_word_id": test_data[1],
                "input_user_bias": test_data[2], "input_difficulty": test_data[3],
                "input_grid_sequence": test_data[4]
            }, batch_size=1024, verbose=1)
            analyze_failure_cases(test_data[5], t_p_steps, t_p_prob, model_name="LSTM (Test)")
            plot_roc_curve(test_data[6], t_p_prob.flatten(), "visualization/LSTM_test_roc_curve.png", "LSTM")
            plot_scatter(test_data[5], np.clip(t_p_steps.flatten(), 0, 7), "visualization/LSTM_test_scatter.png", "LSTM")

    except Exception as e:
        print(f"Skipping LSTM due to error: {e}")

    # -------------------
    # 2. Transformer Model Evaluation
    # -------------------
    try:
        print("\n=== Processing Transformer Model ===")
        from train_transformer import (
            load_tokenizer as load_tf_tok, attach_features as attach_tf_feat,
            build_history as build_tf_hist, TransformerBlock,
            MODEL_SAVE_PATH as TF_PATH, LOOK_BACK as TF_LB,
            focal_loss # <--- [FIX] 必须导入 focal_loss
        )
        
        # [FIX] 在 custom_objects 中加入 focal_loss
        # 注意：这里的 key 必须和训练时保存的名称完全一致：'focal_loss(gamma=2.0,alpha=0.25)'
        tf_model = tf.keras.models.load_model(TF_PATH, custom_objects={
            'TransformerBlock': TransformerBlock,
            'focal_loss(gamma=2.0,alpha=0.25)': focal_loss(alpha=0.25, gamma=2.0)
        })
        tf_tok = load_tf_tok()
        
        val_df = safe_read_csv("dataset/val_data.csv", ["Game", "Trial", "Username", "target", "processed_text"])
        test_df = safe_read_csv("dataset/test_data.csv", ["Game", "Trial", "Username", "target", "processed_text"])
        
        u_df = safe_read_csv("dataset/player_data.csv")
        d_df = safe_read_csv("dataset/difficulty.csv")
        u_map = dict(zip(u_df["Username"], u_df["avg_trial"]))
        d_map = dict(zip(d_df["word"], d_df["avg_trial"]))

        val_df = attach_tf_feat(val_df, tf_tok, u_map, d_map)
        test_df = attach_tf_feat(test_df, tf_tok, u_map, d_map)
        val_hist = build_tf_hist(val_df)
        test_hist = build_tf_hist(test_df)
        
        def create_tf_samples_local(history, look_back):
            from train_transformer import MAX_TRIES, GRID_FEAT_LEN
            X_seq, X_diff, X_wid, X_bias, X_guess, y_st, y_su = [], [], [], [], [], [], []
            for u, evs in history.items():
                if len(evs) <= look_back: continue
                for i in range(look_back, len(evs)):
                    win = evs[i-look_back:i]
                    tgt = evs[i]
                    tr = np.array([t[0] for t in win], np.float32)
                    norm, std = tr/7.0, np.std(tr)/7.0
                    X_seq.append(np.stack([norm, np.full_like(norm, std)], axis=1))
                    X_diff.append([tgt[3]/7.0]) 
                    X_wid.append([tgt[1]])      
                    X_bias.append([tgt[2]/7.0]) 
                    X_guess.append(win[-1][4])  
                    y_st.append(min(float(tgt[0]), 7.0))
                    y_su.append(1.0 if tgt[0] <= 6 else 0.0)
            if not X_seq: return None
            return (np.array(X_seq, np.float32), np.array(X_diff, np.float32), np.array(X_wid, np.int32),
                    np.array(X_bias, np.float32), np.array(X_guess, np.float32), np.array(y_st, np.float32), np.array(y_su, np.float32))

        val_data = create_tf_samples_local(val_hist, TF_LB)
        test_data = create_tf_samples_local(test_hist, TF_LB)
        
        if val_data:
            print("Transformer: Predicting Validation Set...")
            v_p_steps, v_p_prob = tf_model.predict({
                "input_history": val_data[0], "input_difficulty": val_data[1],
                "input_word_id": val_data[2], "input_user_bias": val_data[3],
                "input_guess_sequence": val_data[4]
            }, batch_size=1024, verbose=1)
            analyze_failure_cases(val_data[5], v_p_steps, v_p_prob, model_name="Transformer (Val)")
            plot_roc_curve(val_data[6], v_p_prob.flatten(), "visualization/Transformer_validation_roc_curve.png", "Transformer")
            plot_scatter(val_data[5], np.clip(v_p_steps.flatten(), 0, 7), "visualization/Transformer_validation_scatter.png", "Transformer")
            
        if test_data:
            print("Transformer: Predicting Test Set...")
            t_p_steps, t_p_prob = tf_model.predict({
                "input_history": test_data[0], "input_difficulty": test_data[1],
                "input_word_id": test_data[2], "input_user_bias": test_data[3],
                "input_guess_sequence": test_data[4]
            }, batch_size=1024, verbose=1)
            analyze_failure_cases(test_data[5], t_p_steps, t_p_prob, model_name="Transformer (Test)")
            plot_roc_curve(test_data[6], t_p_prob.flatten(), "visualization/Transformer_test_roc_curve.png", "Transformer")
            plot_scatter(test_data[5], np.clip(t_p_steps.flatten(), 0, 7), "visualization/Transformer_test_scatter.png", "Transformer")

    except Exception as e:
        print(f"Skipping Transformer due to error: {e}")

    print("\nDone. Check 'visualization/' folder for plots.")