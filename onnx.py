import torch
from Stage_SSM import Stage_SSM

# 1. 创建模型实例
model = Stage_SSM(num_class=2)

# 2. 安全加载状态字典
state_dict = torch.load(
    r"C:\Users\74424\PycharmProjects\SSM1\checkpoints\Stage_SSM_all_best_20250804_022049.pth",
    map_location='cpu',  # 明确指定加载到CPU
    weights_only=True    # 安全加载模式，避免警告
)

# 3. 将状态字典加载到模型中
model.load_state_dict(state_dict)
model.eval()  # 设置为评估模式

# 4. 创建正确尺寸的输入张量 - 单通道图像
dummy_input = torch.randn(1, 1, 64, 64, device='cpu')  # 修改为1通道

# 5. 导出ONNX模型
input_names = ["input"]
output_names = ["output"]

try:
    torch.onnx.export(
        model,
        dummy_input,
        "Stage_SSM.onnx",
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=13,

    )
    print("ONNX模型导出成功!")
except Exception as e:
    print(f"导出失败: {e}")
