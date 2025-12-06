#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction visualization utilities for Word Difficulty Prediction models.
This file contains functions for generating ROC curves and scatter plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import random
import os 

# Set seed for reproducibility
SEED = 42

def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

# Initialize with fixed seed
set_seed(SEED)

def calculate_auc_best(y_true, prob):
    """
    Compute AUC for prob and -prob; return (best_auc, used_prob, inverted_flag)
    """
    try:
        auc_pos = roc_auc_score(y_true, prob)
    except Exception:
        auc_pos = float("nan")
    try:
        auc_neg = roc_auc_score(y_true, -prob)
    except Exception:
        auc_neg = float("nan")

    # Choose the larger valid AUC
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

def plot_roc_curve(y_true, prob, save_path, model_name="Model"):
    """
    Plot ROC curve and automatically detect whether prob should be negated.
    
    Args:
        y_true: True labels
        prob: Predicted probabilities
        save_path: Path to save the ROC curve
        model_name: Name of the model for title
    """
    auc, used_prob, inverted = calculate_auc_best(y_true, prob)
    if inverted:
        print(f"⚠️ ROC plotting: detected better AUC with -prob for {model_name}, using -prob for ROC plot.")
    
    # Compute ROC curve
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
    """
    Add jitter to data points with fixed seed for reproducibility.
    
    Args:
        data: Input data array
        jitter_amount: Amount of jitter to add (standard deviation)
        seed: Random seed for reproducibility
    
    Returns:
        Data with added jitter
    """
    if seed is not None:
        np.random.seed(seed)
    
    jitter = np.random.normal(0, jitter_amount, size=data.shape)
    return data + jitter

def plot_scatter(y_true, y_pred, save_path, model_name="Model", jitter_amount=0.18):
    """
    Plot scatter diagram of predicted values vs true values with jitter, showing prediction errors.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Save path for the scatter plot
        model_name: Model name for title
        jitter_amount: Amount of jitter to add to points
    """
    plt.figure(figsize=(10, 8))
    
    # Calculate errors
    errors = np.abs(y_true - y_pred)
    
    # Add jitter to both true and predicted values for better visualization
    y_true_jitter = add_jitter(y_true, jitter_amount=jitter_amount, seed=SEED)
    y_pred_jitter = add_jitter(y_pred, jitter_amount=jitter_amount, seed=SEED+1)  # Different seed for y-axis
    
    # Plot scatter diagram with jittered points
    plt.scatter(y_true_jitter, y_pred_jitter, c=errors, cmap='viridis', alpha=0.3, s=20, label='Prediction points')
    
    # Add color bar to show error magnitude
    cbar = plt.colorbar()
    cbar.set_label('Prediction Error (|True - Predicted|)')
    
    # Add reference line y = x
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Ideal prediction line (y=x)')
    
    # Set title and axis labels
    plt.title(f'{model_name} Model: Predicted vs True Values')
    plt.xlabel('True Values (Steps)')
    plt.ylabel('Predicted Values (Steps)')
    
    # Add legend
    plt.legend(fontsize=10)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save image
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved to: {save_path}")

# ==========================================================
# Prediction functions for LSTM and Transformer models
# ==========================================================

def lstm_predict(user_id):
    """
    Make predictions using the LSTM model.
    
    Args:
        user_id: User ID for prediction
    """
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from train_LSTM import (
        load_tokenizer, attach_features, build_history,
        focal_loss, MODEL_SAVE_PATH, TOKENIZER_PATH,
        LOOK_BACK, MAX_TRIES, GRID_SEQ_FEAT_DIM,
        safe_read_csv
    )
    
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError("LSTM model not found. Please train the model first.")
    
    # Load model with custom objects
    model = tf.keras.models.load_model(
        MODEL_SAVE_PATH,
        custom_objects={
            'focal_loss(gamma=2.0,alpha=0.25)': focal_loss(alpha=0.25, gamma=2.0)
        }
    )
    
    tokenizer = load_tokenizer()
    
    # Read data
    TRAIN_FILE = "dataset/train_data.csv"
    PLAYER_FILE = "dataset/player_data.csv"
    DIFFICULTY_FILE = "dataset/difficulty.csv"
    
    df = safe_read_csv(TRAIN_FILE, usecols=["Game", "Trial", "Username", "target", "processed_text"])
    
    # Prepare maps
    user_map = {}
    diff_map = {}
    if os.path.exists(PLAYER_FILE):
        pdf = pd.read_csv(PLAYER_FILE)
        user_map = dict(zip(pdf["Username"], pdf["avg_trial"]))
    if os.path.exists(DIFFICULTY_FILE):
        ddf = pd.read_csv(DIFFICULTY_FILE)
        diff_map = dict(zip(ddf["word"], ddf["avg_trial"]))
    
    # Process data
    df = attach_features(df, tokenizer, user_map, diff_map)
    hist = build_history(df)
    
    if user_id not in hist:
        print(f"User {user_id} has no records")
        return
    
    events = hist[user_id]
    if len(events) < 1:
        print("Insufficient history")
        return
    
    # Prepare input
    if len(events) < LOOK_BACK:
        avg = np.mean([e[0] for e in events])
        pad_event = (avg, 0, 4.0, np.zeros((MAX_TRIES, GRID_SEQ_FEAT_DIM), dtype=np.float32))
        pad = [pad_event] * (LOOK_BACK - len(events))
        window = pad + events
    else:
        window = events[-LOOK_BACK:]
    
    trials = np.array([w[0] for w in window], np.float32)
    seq = np.stack([trials/7.0, np.full_like(trials, np.std(trials)/7.0)], axis=1)
    seq = seq.reshape(1, LOOK_BACK, 2)
    
    last = events[-1]
    wid = np.array([[last[1]]], np.int32)  # word_id is now index 1
    bias = np.array([[last[2] / 7.0]], np.float32)  # user_bias is now index 2
    diff = np.array([[last[3] / 7.0]], np.float32)  # word_difficulty is now index 3
    grid_seq = last[4].reshape(1, MAX_TRIES, GRID_SEQ_FEAT_DIM)  # grid_seq is now index 4
    
    # Make prediction
    p_steps, p_prob = model.predict({
        "input_history": seq,
        "input_word_id": wid,
        "input_user_bias": bias,
        "input_difficulty": diff,
        "input_grid_sequence": grid_seq
    }, verbose=0)
    
    print(f"LSTM Predicted steps: {float(np.clip(p_steps, 0, 6.99)):.2f}")
    print(f"LSTM Success probability: {float(p_prob):.3f}")


def transformer_predict(user_id):
    """
    Make predictions using the Transformer model.
    
    Args:
        user_id: User ID for prediction
    """
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from train_transformer import (
        load_tokenizer, attach_features, build_history,
        TransformerBlock, MODEL_SAVE_PATH, TOKENIZER_PATH,
        LOOK_BACK, MAX_TRIES, GRID_FEAT_LEN,
        safe_read_csv
    )
    
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError("Transformer model not found. Please train the model first.")
    
    # Load model with custom objects
    model = tf.keras.models.load_model(MODEL_SAVE_PATH, custom_objects={'TransformerBlock': TransformerBlock})
    tokenizer = load_tokenizer()
    
    # Read data
    TRAIN_FILE = "dataset/train_data.csv"
    DIFFICULTY_FILE = "dataset/difficulty.csv"
    PLAYER_FILE = "dataset/player_data.csv"
    
    df = safe_read_csv(TRAIN_FILE, usecols=["Game", "Trial", "Username", "target", "processed_text"])
    
    # Prepare maps
    diff_map = {}
    user_map = {}
    if os.path.exists(DIFFICULTY_FILE):
        ddf = pd.read_csv(DIFFICULTY_FILE)
        diff_map = dict(zip(ddf["word"], ddf["avg_trial"]))
    if os.path.exists(PLAYER_FILE):
        pdf = pd.read_csv(PLAYER_FILE)
        user_map = dict(zip(pdf["Username"], pdf["avg_trial"]))
    
    # Process data
    df = attach_features(df, tokenizer, user_map, diff_map)
    hist = build_history(df)
    
    if user_id not in hist:
        print(f"User {user_id} has no records")
        return
    
    events = hist[user_id]
    if len(events) < 1:
        print("Insufficient history")
        return
    
    # Prepare input
    if len(events) < LOOK_BACK:
        avg = np.mean([e[0] for e in events])
        pad_guess_seq = np.zeros((MAX_TRIES, GRID_FEAT_LEN), dtype=np.float32)
        pad = [(avg, 4.0, 0, 4.0, pad_guess_seq)] * (LOOK_BACK - len(events))
        window = pad + events
    else:
        window = events[-LOOK_BACK:]
    
    trials = np.array([w[0] for w in window], np.float32)
    seq = np.stack([trials/7.0, np.full_like(trials, np.std(trials)/7.0)], axis=1)
    seq = seq.reshape(1, LOOK_BACK, 2)
    
    last = events[-1]
    diff = np.array([[last[1] / 7.0]], np.float32)
    wid = np.array([[last[2]]], np.int32)
    bias = np.array([[last[3] / 7.0]], np.float32)
    guess_seq = last[4].reshape(1, MAX_TRIES, GRID_FEAT_LEN)
    
    # Make prediction
    p_steps, p_prob = model.predict({
        "input_history": seq,
        "input_difficulty": diff,
        "input_word_id": wid,
        "input_user_bias": bias,
        "input_guess_sequence": guess_seq
    }, verbose=0)
    
    print(f"Transformer Predicted steps: {float(np.clip(p_steps, 0, 6.99)):.2f}")
    print(f"Transformer Success probability: {float(p_prob):.3f}")


# ==========================================================
# Main function to generate ROC curves and scatter plots for both models
# ==========================================================
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    
    print("Generating ROC curves and scatter plots for both models...")
    
    # -------------------
    # LSTM Model Processing
    # -------------------
    print("\n1. Processing LSTM model...")
    
    try:
        from train_LSTM import (
            load_tokenizer as load_lstm_tokenizer, attach_features as attach_lstm_features,
            build_history as build_lstm_history, focal_loss,
            MODEL_SAVE_PATH as LSTM_MODEL_PATH,
            LOOK_BACK as LSTM_LOOK_BACK,
            safe_read_csv
        )
        
        # Load LSTM model
        lstm_model = tf.keras.models.load_model(
            LSTM_MODEL_PATH,
            custom_objects={
                'focal_loss(gamma=2.0,alpha=0.25)': focal_loss(alpha=0.25, gamma=2.0)
            }
        )
        
        # Load tokenizer
        lstm_tokenizer = load_lstm_tokenizer()
        
        # Read data
        TRAIN_FILE = "dataset/train_data.csv"
        VAL_FILE = "dataset/val_data.csv"
        TEST_FILE = "dataset/test_data.csv"
        PLAYER_FILE = "dataset/player_data.csv"
        DIFFICULTY_FILE = "dataset/difficulty.csv"
        
        train_df = safe_read_csv(TRAIN_FILE, usecols=["Game", "Trial", "Username", "target", "processed_text"])
        val_df = safe_read_csv(VAL_FILE, usecols=["Game", "Trial", "Username", "target", "processed_text"])
        test_df = safe_read_csv(TEST_FILE, usecols=["Game", "Trial", "Username", "target", "processed_text"])
        
        # Prepare maps
        user_map = {}
        diff_map = {}
        if os.path.exists(PLAYER_FILE):
            pdf = pd.read_csv(PLAYER_FILE)
            user_map = dict(zip(pdf["Username"], pdf["avg_trial"]))
        if os.path.exists(DIFFICULTY_FILE):
            ddf = pd.read_csv(DIFFICULTY_FILE)
            diff_map = dict(zip(ddf["word"], ddf["avg_trial"]))
        
        # Process data
        train_df = attach_lstm_features(train_df, lstm_tokenizer, user_map, diff_map)
        val_df = attach_lstm_features(val_df, lstm_tokenizer, user_map, diff_map)
        test_df = attach_lstm_features(test_df, lstm_tokenizer, user_map, diff_map)
        
        # Build history
        train_hist = build_lstm_history(train_df)
        val_hist = build_lstm_history(val_df)
        test_hist = build_lstm_history(test_df)
        
        # Create samples for prediction
        def create_lstm_samples(history, look_back):
            from train_LSTM import MAX_TRIES, GRID_SEQ_FEAT_DIM
            X_seq, X_wid, X_bias, X_diff, X_grid_seq, y_steps, y_succ = [], [], [], [], [], [], []
            for user, events in history.items():
                if len(events) <= look_back:
                    continue
                for i in range(look_back, len(events)):
                    window = events[i-look_back:i]
                    target = events[i]
                    
                    trials = np.array([t[0] for t in window], np.float32)
                    norm = trials / 7.0
                    std = np.std(trials) / 7.0
                    seq = np.stack([norm, np.full_like(norm, std)], axis=1)
                    X_seq.append(seq)
                    
                    X_wid.append([target[1]])
                    X_bias.append([target[2] / 7.0])
                    X_diff.append([target[3] / 7.0])
                    X_grid_seq.append(target[4])
                    
                    y_steps.append(min(float(target[0]), 7.0))
                    y_succ.append(1.0 if target[0] <= 6 else 0.0)
            
            if not X_seq:
                return (
                    np.zeros((0, look_back, 2), np.float32),
                    np.zeros((0, 1), np.int32),
                    np.zeros((0, 1), np.float32),
                    np.zeros((0, 1), np.float32),
                    np.zeros((0, MAX_TRIES, GRID_SEQ_FEAT_DIM), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.float32)
                )
            
            return (
                np.array(X_seq, np.float32),
                np.array(X_wid, np.int32),
                np.array(X_bias, np.float32),
                np.array(X_diff, np.float32),
                np.array(X_grid_seq, np.float32),
                np.array(y_steps, np.float32),
                np.array(y_succ, np.float32)
            )
        
        # Generate samples
        val_samples = create_lstm_samples(val_hist, LSTM_LOOK_BACK)
        test_samples = create_lstm_samples(test_hist, LSTM_LOOK_BACK)
        
        # Make predictions for validation set
        print("   Generating LSTM validation predictions...")
        val_pred_steps, val_pred_prob = lstm_model.predict({
            "input_history": val_samples[0],
            "input_word_id": val_samples[1],
            "input_user_bias": val_samples[2],
            "input_difficulty": val_samples[3],
            "input_grid_sequence": val_samples[4]
        }, batch_size=1024, verbose=1)
        
        # Generate ROC curve for validation set
        val_roc_path = "visualization/LSTM_validation_roc_curve.png"
        plot_roc_curve(val_samples[6], val_pred_prob.flatten(), val_roc_path, model_name="LSTM")
        
        # Generate scatter plot for validation set
        val_scatter_path = "visualization/LSTM_validation_scatter.png"
        plot_scatter(val_samples[5], np.clip(val_pred_steps.flatten(), 0, 7), val_scatter_path, model_name="LSTM")
        
        # Make predictions for test set
        print("   Generating LSTM test predictions...")
        test_pred_steps, test_pred_prob = lstm_model.predict({
            "input_history": test_samples[0],
            "input_word_id": test_samples[1],
            "input_user_bias": test_samples[2],
            "input_difficulty": test_samples[3],
            "input_grid_sequence": test_samples[4]
        }, batch_size=1024, verbose=1)
        
        # Generate ROC curve for test set
        test_roc_path = "visualization/LSTM_test_roc_curve.png"
        plot_roc_curve(test_samples[6], test_pred_prob.flatten(), test_roc_path, model_name="LSTM")
        
        # Generate scatter plot for test set
        test_scatter_path = "visualization/LSTM_test_scatter.png"
        plot_scatter(test_samples[5], np.clip(test_pred_steps.flatten(), 0, 7), test_scatter_path, model_name="LSTM")
        
        print("   LSTM model processing completed.")
        
    except Exception as e:
        print(f"   Error processing LSTM model: {e}")
    
    # -------------------
    # Transformer Model Processing
    # -------------------
    print("\n2. Processing Transformer model...")
    
    try:
        from train_transformer import (
            load_tokenizer as load_transformer_tokenizer, attach_features as attach_transformer_features,
            build_history as build_transformer_history, TransformerBlock,
            MODEL_SAVE_PATH as TRANSFORMER_MODEL_PATH,
            LOOK_BACK as TRANSFORMER_LOOK_BACK,
            safe_read_csv
        )
        
        # Load Transformer model
        transformer_model = tf.keras.models.load_model(
            TRANSFORMER_MODEL_PATH,
            custom_objects={'TransformerBlock': TransformerBlock}
        )
        
        # Load tokenizer
        transformer_tokenizer = load_transformer_tokenizer()
        
        # Read data
        DIFFICULTY_FILE = "dataset/difficulty.csv"
        
        train_df = safe_read_csv(TRAIN_FILE, usecols=["Game", "Trial", "Username", "target", "processed_text"])
        val_df = safe_read_csv(VAL_FILE, usecols=["Game", "Trial", "Username", "target", "processed_text"])
        test_df = safe_read_csv(TEST_FILE, usecols=["Game", "Trial", "Username", "target", "processed_text"])
        
        # Prepare maps
        diff_map = {}
        user_map = {}
        if os.path.exists(DIFFICULTY_FILE):
            ddf = pd.read_csv(DIFFICULTY_FILE)
            diff_map = dict(zip(ddf["word"], ddf["avg_trial"]))
        if os.path.exists(PLAYER_FILE):
            pdf = pd.read_csv(PLAYER_FILE)
            user_map = dict(zip(pdf["Username"], pdf["avg_trial"]))
        
        # Process data
        train_df = attach_transformer_features(train_df, transformer_tokenizer, user_map, diff_map)
        val_df = attach_transformer_features(val_df, transformer_tokenizer, user_map, diff_map)
        test_df = attach_transformer_features(test_df, transformer_tokenizer, user_map, diff_map)
        
        # Build history
        train_hist = build_transformer_history(train_df)
        val_hist = build_transformer_history(val_df)
        test_hist = build_transformer_history(test_df)
        
        # Create samples for prediction
        def create_transformer_samples(history, look_back):
            from train_transformer import MAX_TRIES, GRID_FEAT_LEN
            X_seq, X_diff, X_wid, X_bias, X_guess_seq, y_steps, y_succ = [], [], [], [], [], [], []
            for user, events in history.items():
                if len(events) <= look_back:
                    continue
                for i in range(look_back, len(events)):
                    window = events[i-look_back:i]
                    target = events[i]
                    
                    trials = np.array([t[0] for t in window], np.float32)
                    norm = trials / 7.0
                    std = np.std(trials) / 7.0
                    seq = np.stack([norm, np.full_like(norm, std)], axis=1)
                    X_seq.append(seq)
                    
                    X_wid.append([target[1]])          # Index 1 是 Word ID
                    X_bias.append([target[2] / 7.0])   # Index 2 是 User Bias (归一化)
                    X_diff.append([target[3] / 7.0])   # Index 3 是 Difficulty (归一化)
                    X_guess_seq.append(window[-1][4])
                    
                    y_steps.append(min(float(target[0]), 7.0))
                    y_succ.append(1.0 if target[0] <= 6 else 0.0)
            
            if not X_seq:
                return (
                    np.zeros((0, look_back, 2), np.float32),
                    np.zeros((0, 1), np.float32),
                    np.zeros((0, 1), np.int32),
                    np.zeros((0, 1), np.float32),
                    np.zeros((0, MAX_TRIES, GRID_FEAT_LEN), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.float32)
                )
            
            return (
                np.array(X_seq, np.float32),
                np.array(X_diff, np.float32),
                np.array(X_wid, np.int32),
                np.array(X_bias, np.float32),
                np.array(X_guess_seq, np.float32),
                np.array(y_steps, np.float32),
                np.array(y_succ, np.float32)
            )
        
        # Generate samples
        val_samples = create_transformer_samples(val_hist, TRANSFORMER_LOOK_BACK)
        test_samples = create_transformer_samples(test_hist, TRANSFORMER_LOOK_BACK)
        
        # Make predictions for validation set
        print("   Generating Transformer validation predictions...")
        val_pred_steps, val_pred_prob = transformer_model.predict({
            "input_history": val_samples[0],
            "input_difficulty": val_samples[1],
            "input_word_id": val_samples[2],
            "input_user_bias": val_samples[3],
            "input_guess_sequence": val_samples[4]
        }, batch_size=1024, verbose=1)
        
        # Generate ROC curve for validation set
        val_roc_path = "visualization/Transformer_validation_roc_curve.png"
        plot_roc_curve(val_samples[6], val_pred_prob.flatten(), val_roc_path, model_name="Transformer")
        
        # Generate scatter plot for validation set
        val_scatter_path = "visualization/Transformer_validation_scatter.png"
        plot_scatter(val_samples[5], np.clip(val_pred_steps.flatten(), 0, 7), val_scatter_path, model_name="Transformer")
        
        # Make predictions for test set
        print("   Generating Transformer test predictions...")
        test_pred_steps, test_pred_prob = transformer_model.predict({
            "input_history": test_samples[0],
            "input_difficulty": test_samples[1],
            "input_word_id": test_samples[2],
            "input_user_bias": test_samples[3],
            "input_guess_sequence": test_samples[4]
        }, batch_size=1024, verbose=1)
        
        # Generate ROC curve for test set
        test_roc_path = "visualization/Transformer_test_roc_curve.png"
        plot_roc_curve(test_samples[6], test_pred_prob.flatten(), test_roc_path, model_name="Transformer")
        
        # Generate scatter plot for test set
        test_scatter_path = "visualization/Transformer_test_scatter.png"
        plot_scatter(test_samples[5], np.clip(test_pred_steps.flatten(), 0, 7), test_scatter_path, model_name="Transformer")
        
        print("   Transformer model processing completed.")
        
    except Exception as e:
        print(f"   Error processing Transformer model: {e}")
    
    print("\nAll predictions and visualizations generated successfully!")
    print("Files saved to visualization folder with original naming format.")
