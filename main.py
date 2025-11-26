# 首先尝试导入所有必要的模块，确保环境完整性
try:
    import os
    import sys
    # 尝试导入主要功能模块，用于验证它们是否存在
    import LSTM_prediction
    import transformer_prediction
except ImportError as e:
    print("❌ 导入模块失败: ", e)
    print("请检查LSTM_prediction.py和transformer_prediction.py文件是否存在")
    sys.exit(1)

def main():
    """Wordle预测系统主程序入口"""
    print("=" * 60)
    print("🎯 Wordle 预测系统 (Wordle Prediction System)")
    print("=" * 60)
    print("该系统支持LSTM和Transformer两种模型的训练和预测功能")
    print("\n请选择要执行的操作:")
    print("  1. 训练模型 (Train Model)")
    print("  2. 使用已有模型进行预测 (Predict with Existing Model)")
    print("  3. 退出 (Exit)")
    print("=" * 60)
    
    # 确保必要的文件夹存在
    os.makedirs('LSTM_Model', exist_ok=True)
    os.makedirs('Transformer_Model', exist_ok=True)
    os.makedirs('visualization', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('dataset', exist_ok=True)  # 确保数据集文件夹存在
    os.makedirs('wandb', exist_ok=True)    # 确保WandB文件夹存在
    
    while True:
        try:
            choice = input("\n请输入您的选择 (1-3): ")
            
            if choice == '1':
                handle_training()
            elif choice == '2':
                handle_prediction()
            elif choice == '3':
                print("\n👋 感谢使用Wordle预测系统，再见!")
                break
            else:
                print("❌ 无效的选择，请重新输入1-3之间的数字")
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，感谢使用!")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            print("建议检查相关文件是否存在，或尝试重新运行程序")

def handle_training():
    """处理模型训练功能"""
    print("\n🚀 进入模型训练模式")
    print("请选择要训练的模型:")
    print("  1. LSTM 模型")
    print("  2. Transformer 模型")
    
    model_choice = input("\n请输入您的选择 (1-2): ")
    
    if model_choice == '1':
        # 检查LSTM_prediction.py是否存在
        if os.path.exists('LSTM_prediction.py'):
            print("\n📊 开始训练LSTM模型...")
            print("注意：训练过程可能需要较长时间，取决于数据量和硬件配置")
            try:
                # 调用LSTM_prediction模块的main函数，设置mode为'train'
                LSTM_prediction.main(mode='train')
            except Exception as e:
                print(f"❌ 执行LSTM_prediction模块过程中发生错误: {e}")
        else:
            print("❌ LSTM_prediction.py文件不存在，请确保所有模块文件已正确创建")
    
    elif model_choice == '2':
        # 检查transformer_prediction.py是否存在
        if os.path.exists('transformer_prediction.py'):
            print("\n📊 开始训练Transformer模型...")
            print("注意：训练过程可能需要较长时间，取决于数据量和硬件配置")
            try:
                # 调用transformer_prediction模块的main函数，设置mode为'train'
                transformer_prediction.main(mode='train')
            except Exception as e:
                print(f"❌ 执行transformer_prediction模块过程中发生错误: {e}")
        else:
            print("❌ transformer_prediction.py文件不存在，请确保所有模块文件已正确创建")
    else:
        print("❌ 无效的选择，请重新输入1-2之间的数字")

def handle_prediction():
    """处理模型预测功能"""
    print("\n🔮 进入预测模式")
    print("请选择要使用的预测模型:")
    print("  1. LSTM 模型")
    print("  2. Transformer 模型")
    
    model_choice = input("\n请输入您的选择 (1-2): ")
    
    # 获取用户ID
    user_id = input("\n请输入要预测的用户ID: ")
    
    # 检查模型是否存在
    if model_choice == '1':
        # 检查LSTM模型文件路径
        lstm_model_path = 'LSTM_Model'  # 根据LSTM_prediction.py中的设置
        if os.path.exists(lstm_model_path) and len(os.listdir(lstm_model_path)) > 0:
            if os.path.exists('LSTM_prediction.py'):
                print("\n📈 使用LSTM模型进行预测...")
                try:
                    # 调用LSTM_prediction模块的main函数，设置mode为'predict'并传入user_id
                    LSTM_prediction.main(mode='predict', user_id=user_id)
                except Exception as e:
                    print(f"❌ 执行LSTM_prediction模块过程中发生错误: {e}")
            else:
                print("❌ LSTM_prediction.py文件不存在")
        else:
            print(f"❌ LSTM模型不存在或未训练，请先训练模型。模型路径: {lstm_model_path}")
    
    elif model_choice == '2':
        # 检查Transformer模型文件路径
        trans_model_path = 'Transformer_Model'  # 根据transformer_prediction.py中的设置
        if os.path.exists(trans_model_path) and len(os.listdir(trans_model_path)) > 0:
            if os.path.exists('transformer_prediction.py'):
                print("\n📈 使用Transformer模型进行预测...")
                try:
                    # 调用transformer_prediction模块的main函数，设置mode为'predict'并传入user_id
                    transformer_prediction.main(mode='predict', user_id=user_id)
                except Exception as e:
                    print(f"❌ 执行transformer_prediction模块过程中发生错误: {e}")
            else:
                print("❌ transformer_prediction.py文件不存在")
        else:
            print(f"❌ Transformer模型不存在或未训练，请先训练模型。模型路径: {trans_model_path}")
    else:
        print("❌ 无效的选择，请重新输入1-2之间的数字")

if __name__ == "__main__":
    main()