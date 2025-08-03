import os
from openvino.runtime import serialize
from openvino.runtime import Core

# 定义 ONNX 模型路径
onnx_model_path = r"C:\Users\74424\PycharmProjects\SSM1\Stage_SSM.onnx"

# 定义输出目录路径
output_dir = r"C:\Users\74424\PycharmProjects\SSM1"

# 检查输出目录是否存在，如果不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory at: {output_dir}")
else:
    print(f"Output directory already exists at: {output_dir}")

# 使用 OpenVINO Core API 进行转换
try:
    # 创建 OpenVINO Core 对象
    core = Core()

    # 读取 ONNX 模型
    model = core.read_model(onnx_model_path)

    # 序列化为 OpenVINO IR 格式
    xml_path = os.path.join(output_dir, "model.xml")
    bin_path = os.path.join(output_dir, "model.bin")
    serialize(model, xml_path, bin_path)

    print(f"Model successfully converted and saved to: {xml_path} and {bin_path}")
except Exception as e:
    print(f"Error during model conversion: {e}")