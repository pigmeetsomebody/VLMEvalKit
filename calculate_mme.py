import pandas as pd
import os

def calculate_mme_score(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 '{file_path}'")
        return

    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 检查必要的列是否存在
        required_columns = ['perception', 'reasoning']
        if not all(col in df.columns for col in required_columns):
            print(f"错误: CSV文件中缺少必要的列: {required_columns}")
            return

        # 获取第一行数据（假设只有一行分数）
        row = df.iloc[0]
        
        perception_score = row['perception']
        reasoning_score = row['reasoning']
        
        # MME总分 = Perception + Reasoning
        total_score = perception_score + reasoning_score
        
        print(f"读取文件: {file_path}")
        print("-" * 40)
        print(f"Perception (感知) 分数: {perception_score}")
        print(f"Reasoning  (推理) 分数: {reasoning_score}")
        print("-" * 40)
        print(f"MME 总分: {total_score}")
        
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 定义文件名
    csv_filename = '/home/zhuyy/VLMEvalKit/outputs_16_scalars/Qwen2.5-VL-7B-Instruct/Qwen2.5-VL-7B-Instruct_MME_score.csv'
    
    # 调用函数
    calculate_mme_score(csv_filename)