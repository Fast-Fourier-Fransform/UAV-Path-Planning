import scipy.io as sio

def load_mat_data(filename, var_name):
    """
    读取 .mat 文件并提取对应变量的 1D 数组。
    如果不知道变量名，可以在报错时打印 data.keys() 检查。
    """
    try:
        data = sio.loadmat(filename)
        return data[var_name].flatten()
    except FileNotFoundError:
        print(f"警告: 未找到文件 {filename}，跳过加载。")
        return None
    except KeyError:
        print(f"警告: 在 {filename} 中未找到变量 '{var_name}'。")
        # print("可用变量:", data.keys()) # 调试时可取消注释
        return None