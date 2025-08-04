import numpy as np
import cv2
from skimage.measure import label, regionprops
from openvino.runtime import Core


class SegmentationInference:
    def __init__(self, model_xml_path, device="CPU"):
        """
        初始化OpenVINO分割模型

        参数:
        model_xml_path: str, OpenVINO模型路径(.xml文件)
        device: str, 推理设备 (默认: "CPU")
        """
        # 加载OpenVINO模型
        self.core = Core()
        self.model = self.core.read_model(model_xml_path)
        self.compiled_model = self.core.compile_model(self.model, device)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        # 获取输入尺寸 (N,C,H,W)
        self.input_shape = self.input_layer.shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def preprocess(self, image):
        """
        图像预处理: 调整大小, 归一化, 转换维度

        参数:
        image: numpy数组, 输入图像 (H,W,3)

        返回:
        processed_image: numpy数组, 预处理后的图像 (1,3,H,W)
        """
        # 调整大小并保持宽高比
        orig_h, orig_w = image.shape[:2]
        scale = min(self.input_height / orig_h, self.input_width / orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 填充到模型输入尺寸
        padded = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        padded[:new_h, :new_w, :] = resized

        # 归一化并转换维度
        normalized = padded.astype(np.float32) / 255.0
        return np.expand_dims(normalized.transpose(2, 0, 1), 0)  # HWC -> CHW -> NCHW

    def postprocess(self, output, orig_shape, threshold=0.5):
        """
        后处理: 获取二值掩码, 保留最大连通区域

        参数:
        output: numpy数组, 模型输出 (1,1,H,W)
        orig_shape: tuple, 原始图像尺寸 (H,W,C)
        threshold: float, 二值化阈值 (默认: 0.5)

        返回:
        final_mask: numpy数组, 后处理后的二值掩码 (原始尺寸)
        """
        # 1. 获取概率图并调整到原始尺寸
        prob_map = output[0, 0]  # 取出单通道概率图
        prob_map = cv2.resize(prob_map, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)

        # 2. 二值化
        _, binary_mask = cv2.threshold(prob_map, threshold, 1, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)

        # 3. 保留最大连通区域
        return self.keep_largest_region(binary_mask)

    @staticmethod
    def keep_largest_region(mask):
        """
        保留二值分割结果中的最大连通区域

        参数:
        mask: numpy数组, 二值分割结果 (0为背景, 1为前景)

        返回:
        largest_region: numpy数组, 只包含最大连通区域的二值图像
        """
        if mask.max() == 0:
            return mask  # 全背景图像直接返回

        # 标记连通区域
        labeled_mask = label(mask, connectivity=2)
        regions = regionprops(labeled_mask)

        if not regions:
            return mask

        # 找到面积最大的区域
        largest_region = max(regions, key=lambda r: r.area)
        result_mask = np.zeros_like(mask)
        result_mask[labeled_mask == largest_region.label] = 1

        return result_mask

    def predict(self, image_path, threshold=0.5):
        """
        完整的预测流程

        参数:
        image_path: str, 输入图像路径
        threshold: float, 二值化阈值

        返回:
        original_image: 原始图像
        final_mask: 后处理后的二值掩码
        """
        # 1. 读取图像
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        orig_shape = original_image.shape

        # 2. 预处理
        input_tensor = self.preprocess(original_image)

        # 3. OpenVINO推理
        output = self.compiled_model([input_tensor])[self.output_layer]

        # 4. 后处理
        final_mask = self.postprocess(output, orig_shape, threshold)

        return original_image, final_mask


# 使用示例
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 初始化模型
    model_xml = "your_model_path.xml"  # 替换为实际模型路径
    segmenter = SegmentationInference(model_xml)

    # 进行预测
    image_path = "test_image.jpg"  # 替换为测试图像路径
    original_image, result_mask = segmenter.predict(image_path, threshold=0.5)

    # 可视化结果
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(result_mask, cmap='gray')
    axes[1].set_title('Segmentation Result')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()