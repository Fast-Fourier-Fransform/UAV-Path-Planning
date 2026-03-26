import numpy as np
import opfunu

def get_cec_benchmark(year, func_num, dim=30):
    """
    通用 CEC 测试函数获取接口，带自动维度自适应保护
    """
    try:
        module = getattr(opfunu.cec_based, f"cec{year}")
        func_class = getattr(module, f"F{func_num}{year}")
    except AttributeError:
        raise ValueError(f"opfunu 库中未找到 CEC{year} 的 F{func_num} 函数，请检查年份或编号！")

    if year == 2019:
        cec_func = func_class()
        actual_dim = cec_func.ndim
    else:
        try:
            # 1. 优先尝试使用你要求的维度 (比如 30维)
            cec_func = func_class(ndim=dim)
            actual_dim = dim
        except ValueError as e:
            # 2. 如果官方库报错 (如 CEC2022 F9 最多支持 20维)，自动捕获并回退
            print(f"  [提示] 触发官方限制 '{e}' -> 已自动回退至该函数支持的默认/最大维度！")
            cec_func = func_class()  # 不传参，使用该函数内置的默认合法维度
            actual_dim = getattr(cec_func, 'ndim', getattr(cec_func, 'dim', dim))
        except TypeError:
            # 兼容极少数不接受传参的远古函数
            cec_func = func_class()
            actual_dim = getattr(cec_func, 'ndim', getattr(cec_func, 'dim', dim))

    lb_arr = cec_func.lb
    ub_arr = cec_func.ub

    # 包装为算法可调用的适应度函数 (支持多粒子矩阵化评估)
    def cost_func(x):
        if x.ndim == 1:
            return cec_func.evaluate(x)
        else:
            return np.array([cec_func.evaluate(xi) for xi in x])

    return cost_func, lb_arr, ub_arr, actual_dim