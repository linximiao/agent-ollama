import pandas as pd
import matplotlib.pyplot as plt
import json

def describe_data(filepath:str) -> str:
    """
    生成路径为filepath的csv文件的统计描述
    """
    try:
        df = pd.read_csv(filepath)
        result = {
                "行数": len(df),
                "列数": len(df.columns),
                "每列数据类型": [{str(key):str(ty)} for key,ty in zip(df.dtypes.keys(),df.dtypes)],
                "数值列统计": df.select_dtypes(include=['number']).describe().to_dict(),
                "缺失值统计": df.isnull().sum().to_dict()
            }
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f'分析失败：{e}'

def handle_missing(df):
    """
    处理缺失值（填充0）

    参数:
        df (pd.DataFrame): 输入数据

    返回:
        pd.DataFrame: 处理后的数据
    """
    return df.fillna(0)


def plot_bar(df, x_col, y_col, title):
    """
    绘制柱状图

    参数:
        df (pd.DataFrame): 输入数据
        x_col (str): x轴列名
        y_col (str): y轴列名
        title (str): 图表标题
    """
    plt.figure(figsize=(10,6))
    plt.bar(df[x_col], df[y_col])
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()


def save_results(df, output_path):
    """
    将处理后的数据保存到CSV文件

    参数:
        df (pd.DataFrame): 输入数据
        output_path (str): 输出文件路径
    """
    df.to_csv(output_path, index=False)


# 示例用法
if __name__ == "__main__":
    df = pd.read_csv("test/sample.csv")
    print("原始数据:")
    print(df.head())
    
    stats = describe_data(df)
    print("\n统计信息:")
    print(stats)
    
    cleaned_df = handle_missing(df)
    print("\n处理后数据:")
    print(cleaned_df.head())
    
    save_results(cleaned_df, "test/cleaned_data.csv")
    
    # 数据可视化示例
    plot_bar(cleaned_df, "Category", "Values", "数据分布柱状图")