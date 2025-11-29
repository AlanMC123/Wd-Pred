import tensorflow as tf
import numpy as np

# 导入两个模型文件中的Focal Loss函数
tf.compat.v1.disable_eager_execution()

# 模拟LSTM_prediction.py中的Focal Loss实现
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    自定义Focal Loss实现，用于处理类别不平衡和难易样本问题
    
    Args:
        y_true: 真实标签
        y_pred: 预测概率
        gamma: 聚焦参数，控制对难易样本的关注程度，通常为2.0
        alpha: 平衡因子，控制正负样本的权重，通常为0.25
    
    Returns:
        Focal Loss值
    """
    # 确保y_pred在有效范围内，避免log(0)或log(1)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    # 计算交叉熵
    ce = y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
    
    # 计算focal loss的调节因子
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    
    # 计算focal loss
    focal_loss = -alpha_t * tf.math.pow((1 - p_t), gamma) * ce
    
    return tf.reduce_mean(focal_loss)

# 测试函数
def test_focal_loss():
    print("===== 测试Focal Loss实现 =====")
    
    # 测试场景1：正确分类的简单样本
    y_true1 = tf.constant([[1.0], [0.0]], dtype=tf.float32)
    y_pred1 = tf.constant([[0.95], [0.05]], dtype=tf.float32)
    
    # 测试场景2：错误分类的困难样本
    y_true2 = tf.constant([[1.0], [0.0]], dtype=tf.float32)
    y_pred2 = tf.constant([[0.4], [0.6]], dtype=tf.float32)
    
    # 测试场景3：类别不平衡（多数负样本）
    y_true3 = tf.constant([[1.0], [0.0], [0.0], [0.0]], dtype=tf.float32)
    y_pred3 = tf.constant([[0.8], [0.1], [0.2], [0.15]], dtype=tf.float32)
    
    # 测试场景4：边界预测（概率接近0.5）
    y_true4 = tf.constant([[1.0], [0.0]], dtype=tf.float32)
    y_pred4 = tf.constant([[0.55], [0.45]], dtype=tf.float32)
    
    # 计算不同gamma值的影响
    for gamma in [0.0, 1.0, 2.0, 3.0]:
        print(f"\n使用gamma={gamma}:")
        loss1 = focal_loss(y_true1, y_pred1, gamma=gamma)
        loss2 = focal_loss(y_true2, y_pred2, gamma=gamma)
        
        # 启动会话计算
        with tf.compat.v1.Session() as sess:
            loss1_val = sess.run(loss1)
            loss2_val = sess.run(loss2)
            
            print(f"  简单样本损失: {loss1_val:.6f}")
            print(f"  困难样本损失: {loss2_val:.6f}")
            print(f"  难易样本权重比: {loss2_val/loss1_val:.2f} 倍")
    
    # 计算不同alpha值的影响（gamma固定为2.0）
    print("\n===== 测试不同alpha值的影响 (gamma=2.0) =====")
    for alpha in [0.25, 0.5, 0.75]:
        print(f"\n使用alpha={alpha}:")
        loss_pos = focal_loss(tf.constant([[1.0]], dtype=tf.float32), 
                             tf.constant([[0.6]], dtype=tf.float32), 
                             gamma=2.0, alpha=alpha)
        loss_neg = focal_loss(tf.constant([[0.0]], dtype=tf.float32), 
                             tf.constant([[0.4]], dtype=tf.float32), 
                             gamma=2.0, alpha=alpha)
        
        with tf.compat.v1.Session() as sess:
            loss_pos_val = sess.run(loss_pos)
            loss_neg_val = sess.run(loss_neg)
            
            print(f"  正样本损失权重: {loss_pos_val:.6f}")
            print(f"  负样本损失权重: {loss_neg_val:.6f}")
    
    # 与标准交叉熵对比
    print("\n===== 与标准交叉熵对比 =====")
    # 标准二元交叉熵
    def binary_crossentropy(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        return tf.reduce_mean(-(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)))
    
    # 计算不同样本的CE和Focal Loss
    for i, (y_true, y_pred, name) in enumerate([
        (y_true1, y_pred1, "简单样本"),
        (y_true2, y_pred2, "困难样本"),
        (y_true3, y_pred3, "类别不平衡样本"),
        (y_true4, y_pred4, "边界预测样本")
    ]):
        ce_loss = binary_crossentropy(y_true, y_pred)
        focal_loss_val = focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25)
        
        with tf.compat.v1.Session() as sess:
            ce_val, focal_val = sess.run([ce_loss, focal_loss_val])
            
            print(f"\n{name}:")
            print(f"  交叉熵损失: {ce_val:.6f}")
            print(f"  Focal Loss: {focal_val:.6f}")
            print(f"  Focal Loss 对CE的比例: {focal_val/ce_val:.4f}")

# 测试边界情况
def test_edge_cases():
    print("\n===== 测试边界情况 =====")
    
    # 测试极限预测值（接近0或1）
    y_true = tf.constant([[1.0], [0.0]], dtype=tf.float32)
    y_pred_extreme = tf.constant([[0.9999], [0.0001]], dtype=tf.float32)
    
    loss = focal_loss(y_true, y_pred_extreme)
    
    with tf.compat.v1.Session() as sess:
        loss_val = sess.run(loss)
        print(f"极端预测值的Focal Loss: {loss_val:.6f}")
        print(f"(确保没有数值溢出问题)")

if __name__ == "__main__":
    test_focal_loss()
    test_edge_cases()
    
    print("\n✅ Focal Loss 测试完成")
    print("\n在模型中的效果：")
    print("- gamma参数控制难易样本的关注度，gamma越大，模型越关注困难样本")
    print("- alpha参数控制正负样本的权重，当正样本较少时可设置较小的alpha值")
    print("- 在实际训练中，建议先使用默认参数(gamma=2.0, alpha=0.25)，再根据验证效果调整")
