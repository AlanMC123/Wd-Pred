import os
from graphviz import Digraph

# ==========================================
# 配置输出目录
# ==========================================
OUTPUT_DIR = 'structure'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Images will be saved to: ./{OUTPUT_DIR}/")

# ==========================================
# 1. LSTM 模型配置
# ==========================================
LSTM_CONFIG = {
    "LSTM_UNITS": 56,
    "DROPOUT_RATE": 0.45,
    "EMBEDDING_DIM": 24,
    "L2_REG": 0.001,
    "GRID_LSTM_UNITS": 14,
    "DENSE_1": 16,
    "MERGE_DENSE": 64,
    "STEPS_HEAD_HIDDEN": 32,
    "SUCC_HEAD_HIDDEN_1": 64,
    "SUCC_HEAD_DROP_1": 0.45,
    "SUCC_HEAD_HIDDEN_2": 32
}

# ==========================================
# 2. Transformer 模型配置
# ==========================================
TRANS_CONFIG = {
    "PROJECT_DIM": 16,
    "NUM_HEADS": 6,
    "FF_DIM": 16,
    "LAYERS": 1,
    "DROPOUT_RATE": 0.45,
    "EMBEDDING_DIM": 16,
    "MERGE_DENSE": 64,
    "STEPS_HEAD_HIDDEN": 32,
    "STEPS_HEAD_DROP": 0.2,
    "SUCC_HEAD_HIDDEN_1": 64,
    "SUCC_HEAD_DROP_1": 0.3,
    "SUCC_HEAD_HIDDEN_2": 32,
    "SUCC_HEAD_DROP_2": 0.2
}

# [修复] 添加 **kwargs 以接收 fontsize 等额外参数
def create_node(dot, name, label, shape='box', style='filled', color='white', fillcolor='white', **kwargs):
    dot.node(name, label, shape=shape, style=style, color=color, fillcolor=fillcolor, **kwargs)

def draw_lstm_model():
    dot = Digraph('LSTM_Network', comment='LSTM Architecture')
    
    # [调整] LR 布局 (左到右)，扁平化
    # 增加 nodesep/ranksep 防止大字体导致的重叠
    dot.attr(rankdir='LR', splines='ortho', nodesep='0.4', ranksep='0.5')
    
    # [调整] 全局字体大小调大至 16
    dot.attr('node', fontname='Helvetica', fontsize='16') 
    dot.attr('edge', fontsize='12') # 连线标签字体

    # --- 输入层 ---
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='Inputs', style='dashed', color='gray', fontsize='18')
        c.attr(rank='same') 
        create_node(c, 'h_in', 'History Input\n(5, 2)', fillcolor='#E1F5FE')
        create_node(c, 'w_in', 'Word ID\n(1,)', fillcolor='#E1F5FE')
        create_node(c, 'u_in', 'User Bias\n(1,)', fillcolor='#E1F5FE')
        create_node(c, 'd_in', 'Difficulty\n(1,)', fillcolor='#E1F5FE')
        create_node(c, 'g_in', 'Grid Sequence\n(6, 8)', fillcolor='#E1F5FE')

    # --- 分支处理 ---
    # History Branch
    create_node(dot, 'lstm_h', f"LSTM\nUnits: {LSTM_CONFIG['LSTM_UNITS']}", fillcolor='#FFF9C4')
    create_node(dot, 'drop_h', f"Drop\n{LSTM_CONFIG['DROPOUT_RATE']}", shape='ellipse', fillcolor='#FFCCBC')
    dot.edge('h_in', 'lstm_h')
    dot.edge('lstm_h', 'drop_h')

    # Word Branch
    create_node(dot, 'emb_w', f"Embed\n{LSTM_CONFIG['EMBEDDING_DIM']}", fillcolor='#C8E6C9')
    create_node(dot, 'flat_w', "Flatten", fillcolor='#F0F4C3')
    dot.edge('w_in', 'emb_w')
    dot.edge('emb_w', 'flat_w')

    # User Bias Branch
    create_node(dot, 'dense_u', f"Dense\n16 (L2)", fillcolor='#F0F4C3')
    dot.edge('u_in', 'dense_u')

    # Difficulty Branch
    create_node(dot, 'dense_d', f"Dense\n16 (L2)", fillcolor='#F0F4C3')
    dot.edge('d_in', 'dense_d')

    # Grid Sequence Branch
    create_node(dot, 'lstm_g', f"LSTM\n{LSTM_CONFIG['GRID_LSTM_UNITS']}", fillcolor='#FFF9C4')
    create_node(dot, 'drop_g', f"Drop\n{LSTM_CONFIG['DROPOUT_RATE']}", shape='ellipse', fillcolor='#FFCCBC')
    create_node(dot, 'dense_g', f"Dense\n16 (L2)", fillcolor='#F0F4C3')
    dot.edge('g_in', 'lstm_g')
    dot.edge('lstm_g', 'drop_g')
    dot.edge('drop_g', 'dense_g')

    # --- 合并层 ---
    create_node(dot, 'concat', 'Concatenate', shape='invtrapezium', fillcolor='#E0E0E0', style='filled, dashed')
    dot.edge('drop_h', 'concat')
    dot.edge('flat_w', 'concat')
    dot.edge('dense_u', 'concat')
    dot.edge('dense_d', 'concat')
    dot.edge('dense_g', 'concat')

    # --- 共享特征提取 ---
    create_node(dot, 'shared_dense', f"Dense\n{LSTM_CONFIG['MERGE_DENSE']}", fillcolor='#D1C4E9')
    create_node(dot, 'shared_drop', f"Drop\n{LSTM_CONFIG['DROPOUT_RATE']}", shape='ellipse', fillcolor='#FFCCBC')
    dot.edge('concat', 'shared_dense')
    dot.edge('shared_dense', 'shared_drop')

    # --- 输出头 (Heads) ---
    # Steps
    with dot.subgraph(name='cluster_steps') as c:
        c.attr(label='Steps Head', style='dotted', fontsize='18')
        create_node(c, 'step_dense', f"Dense {LSTM_CONFIG['STEPS_HEAD_HIDDEN']}", fillcolor='#B2DFDB')
        create_node(c, 'out_steps', "Out: Steps\n(Linear)", shape='doublecircle', fillcolor='#80CBC4')
        dot.edge('shared_drop', 'step_dense')
        dot.edge('step_dense', 'out_steps')

    # Success
    with dot.subgraph(name='cluster_succ') as c:
        c.attr(label='Success Head', style='dotted', fontsize='18')
        create_node(c, 'succ_dense1', f"Dense {LSTM_CONFIG['SUCC_HEAD_HIDDEN_1']}", fillcolor='#FFCC80')
        create_node(c, 'succ_drop1', f"Drop {LSTM_CONFIG['SUCC_HEAD_DROP_1']}", shape='ellipse', fillcolor='#FFAB91')
        create_node(c, 'succ_dense2', f"Dense {LSTM_CONFIG['SUCC_HEAD_HIDDEN_2']}", fillcolor='#FFCC80')
        create_node(c, 'out_succ', "Out: Success\n(Sigmoid)", shape='doublecircle', fillcolor='#FFAB91')
        
        dot.edge('shared_drop', 'succ_dense1')
        dot.edge('succ_dense1', 'succ_drop1')
        dot.edge('succ_drop1', 'succ_dense2')
        dot.edge('succ_dense2', 'out_succ')

    return dot

def draw_transformer_model():
    dot = Digraph('Transformer_Network', comment='Transformer Architecture')
    
    # [调整] LR 布局，扁平化
    dot.attr(rankdir='LR', splines='ortho', nodesep='0.4', ranksep='0.5')
    # [调整] 全局字体大小调大至 16
    dot.attr('node', fontname='Helvetica', fontsize='16')
    dot.attr('edge', fontsize='12')

    # --- 输入层 ---
    with dot.subgraph(name='cluster_inputs_t') as c:
        c.attr(label='Inputs', style='dashed', color='gray', fontsize='18')
        c.attr(rank='same') # 强制对齐
        create_node(c, 'h_in', 'History\n(5, 2)', fillcolor='#E1F5FE')
        create_node(c, 'g_in', 'Guess Seq\n(6, 8)', fillcolor='#E1F5FE')
        create_node(c, 'w_in', 'Word ID\n(1,)', fillcolor='#E1F5FE')
        create_node(c, 'u_in', 'User Bias\n(1,)', fillcolor='#E1F5FE')
        create_node(c, 'd_in', 'Difficulty\n(1,)', fillcolor='#E1F5FE')

    # --- Transformer 分支 ---
    for branch, input_node in [('history', 'h_in'), ('guess', 'g_in')]:
        p = branch
        create_node(dot, f'{p}_proj', f"Proj Dense\n{TRANS_CONFIG['PROJECT_DIM']}", fillcolor='#E1BEE7')
        # [调整] 这里的 fontsize 显式设置为 12 (比全局稍小一点，因为是辅助节点)
        create_node(dot, f'{p}_pos', f"+ Pos Emb", shape='circle', fillcolor='#E1BEE7', fontsize='12')
        create_node(dot, f'{p}_trans', f"Transformer x{TRANS_CONFIG['LAYERS']}\nHeads:{TRANS_CONFIG['NUM_HEADS']}", shape='component', fillcolor='#D1C4E9')
        create_node(dot, f'{p}_gap', "Avg Pool", fillcolor='#B39DDB')
        create_node(dot, f'{p}_drop', f"Drop\n{TRANS_CONFIG['DROPOUT_RATE']}", shape='ellipse', fillcolor='#FFCCBC')
        
        dot.edge(input_node, f'{p}_proj')
        dot.edge(f'{p}_proj', f'{p}_pos')
        dot.edge(f'{p}_pos', f'{p}_trans')
        dot.edge(f'{p}_trans', f'{p}_gap')
        dot.edge(f'{p}_gap', f'{p}_drop')

    # --- 辅助分支 ---
    create_node(dot, 'emb_w', f"Embed\n{TRANS_CONFIG['EMBEDDING_DIM']}", fillcolor='#C8E6C9')
    create_node(dot, 'flat_w', "Flatten", fillcolor='#F0F4C3')
    dot.edge('w_in', 'emb_w')
    dot.edge('emb_w', 'flat_w')

    create_node(dot, 'dense_u', "Dense\n16", fillcolor='#F0F4C3')
    dot.edge('u_in', 'dense_u')

    create_node(dot, 'dense_d', "Dense\n16", fillcolor='#F0F4C3')
    dot.edge('d_in', 'dense_d')

    # --- 合并 ---
    create_node(dot, 'concat', 'Concatenate', shape='invtrapezium', fillcolor='#E0E0E0')
    dot.edge('history_drop', 'concat')
    dot.edge('flat_w', 'concat')
    dot.edge('dense_u', 'concat')
    dot.edge('dense_d', 'concat')
    dot.edge('guess_drop', 'concat')

    # --- 共享层 ---
    create_node(dot, 'shared_dense', f"Dense\n{TRANS_CONFIG['MERGE_DENSE']}", fillcolor='#9575CD')
    create_node(dot, 'shared_drop', f"Drop\n{TRANS_CONFIG['DROPOUT_RATE']}", shape='ellipse', fillcolor='#FFCCBC')
    dot.edge('concat', 'shared_dense')
    dot.edge('shared_dense', 'shared_drop')

    # --- Heads ---
    # Steps
    with dot.subgraph(name='cluster_steps_t') as c:
        c.attr(label='Steps Head', style='dotted', fontsize='18')
        create_node(c, 'step_dense', f"Dense {TRANS_CONFIG['STEPS_HEAD_HIDDEN']}", fillcolor='#B2DFDB')
        create_node(c, 'step_drop', f"Drop {TRANS_CONFIG['STEPS_HEAD_DROP']}", shape='ellipse', fillcolor='#FFAB91')
        create_node(c, 'out_steps', "Out: Steps", shape='doublecircle', fillcolor='#80CBC4')
        
        dot.edge('shared_drop', 'step_dense')
        dot.edge('step_dense', 'step_drop')
        dot.edge('step_drop', 'out_steps')

    # Success
    with dot.subgraph(name='cluster_succ_t') as c:
        c.attr(label='Success Head', style='dotted', fontsize='18')
        create_node(c, 'succ_dense1', f"Dense {TRANS_CONFIG['SUCC_HEAD_HIDDEN_1']}", fillcolor='#FFCC80')
        create_node(c, 'succ_drop1', f"Drop {TRANS_CONFIG['SUCC_HEAD_DROP_1']}", shape='ellipse', fillcolor='#FFAB91')
        create_node(c, 'succ_dense2', f"Dense {TRANS_CONFIG['SUCC_HEAD_HIDDEN_2']}", fillcolor='#FFCC80')
        create_node(c, 'succ_drop2', f"Drop {TRANS_CONFIG['SUCC_HEAD_DROP_2']}", shape='ellipse', fillcolor='#FFAB91')
        create_node(c, 'out_succ', "Out: Success", shape='doublecircle', fillcolor='#FFAB91')
        
        dot.edge('shared_drop', 'succ_dense1')
        dot.edge('succ_dense1', 'succ_drop1')
        dot.edge('succ_drop1', 'succ_dense2')
        dot.edge('succ_dense2', 'succ_drop2')
        dot.edge('succ_drop2', 'out_succ')

    return dot

if __name__ == '__main__':
    print("Generating LSTM Architecture Diagram (LR Layout)...")
    lstm_dot = draw_lstm_model()
    lstm_output_path = os.path.join(OUTPUT_DIR, 'lstm_architecture_flat')
    lstm_dot.render(lstm_output_path, view=False, format='png', cleanup=True)
    print(f"Saved: {lstm_output_path}.png")

    print("Generating Transformer Architecture Diagram (LR Layout)...")
    trans_dot = draw_transformer_model()
    trans_output_path = os.path.join(OUTPUT_DIR, 'transformer_architecture_flat')
    trans_dot.render(trans_output_path, view=False, format='png', cleanup=True)
    print(f"Saved: {trans_output_path}.png")