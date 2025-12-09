from data_provider.data_factory import data_provider
from argparse import Namespace

# 创建最简配置
args = Namespace(
    data='ETTh1',
    root_path='./dataset/ETT-small/',
    data_path='ETTh1.csv',
    features='M',  # M: 多变量, S: 单变量
    target='OT',
    seq_len=96,
    label_len=48,
    pred_len=96,
    batch_size=32,
    freq='h',
    embed='timeF',
    num_workers=0,
    seasonal_patterns=None,
    augmentation_ratio=0
)

# 加载训练数据
data_set, data_loader = data_provider(args, 'train')

# 获取一个批次数据
batch = next(iter(data_loader))
batch_x, batch_y, batch_x_mark, batch_y_mark = batch

# 输出数据信息
print(f"数据集大小: {len(data_set)}")
print(f"\n批次数据形状:")
print(f"  batch_x (输入序列): {batch_x.shape}")
print(f"  batch_y (目标序列): {batch_y.shape}")
print(f"  batch_x_mark (输入时间特征): {batch_x_mark.shape}")
print(f"  batch_y_mark (目标时间特征): {batch_y_mark.shape}")

print(f"\n第一个样本的前5个时间步:")
print(batch_x[0][:5])

