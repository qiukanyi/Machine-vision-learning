#使用INT8量化减少模型大小并提升推理速度。
# 技术要点
# 硬件兼容性：量化模型需适配目标硬件（如NVIDIA Jetson）。
# 精度损失：通过量化感知训练（QAT）平衡精度与速度。

#需要先安装PyTorch和TorchVision
#pip install torch torchvision torchaudio   --index-url https://pypi.tuna.tsinghua.edu.cn/simple


#示例：PyTorch模型量化
import torch
import os  # 新增文件操作依赖
import yaml
import logging

class QuantConfig:
    def __init__(self, config_path='quant_config.yaml'):
        self._load_config(config_path)

    def _load_config(self, path):
        try:
            with open(path) as f:
                config = yaml.safe_load(f)
                self.model_path = config.get('model_path', 'model.pth')
                self.calibration_samples = config.get('calibration_samples', 32)
                self.input_shape = tuple(config.get('input_shape', [1,512,512]))
        except FileNotFoundError:
            logging.warning("配置文件缺失，使用默认参数") 

#加载模型
model = torch.load("model.pth")

#设置量化配置
model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    
#准备量化
quantized_model = torch.quantization.prepare(model)

# 量化校准流程(优化版)
quantized_model.eval()
# 生成更具代表性的校准数据（模拟实际医疗影像输入）
calibration_data = torch.randn(32, 1, 512, 512)  # 单通道512x512医学影像
for _ in range(3):  # 多轮校准提升精度
    with torch.no_grad():
        _ = quantized_model(calibration_data)

#增加量化校准流程
quantized_model.eval()
# 使用代表性校准数据集（示例）
calibration_data = torch.randn(32, 1, 512, 512)  # 假设输入尺寸为256x256
with torch.no_grad():
    _ = quantized_model(calibration_data)

# 增加量化效果验证（添加在转换之后）
print(f"量化后模型大小：{os.path.getsize('quantized_unet.pth')/1024:.1f} KB")
print(f"原始模型大小：{os.path.getsize('model.pth')/1024:.1f} KB")

#转换并且保存
# 可配置的量化流程
def quantize_model(model, config):
    logging.info("开始模型量化")
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    prepared = torch.quantization.prepare(model)
    
    # 执行校准
    with torch.no_grad():
        prepared(calibration_data)
    
    return torch.quantization.convert(prepared)

quantized_model = quantize_model(model, config)
logging.info("量化完成")

# 自动化测试入口
if __name__ == "__main__":
    pytest.main(['-v', 'test_quantization.py'])
torch.save(quantized_model.state_dict(), "quantized_unet.pth")

# 模型大小对比
import os
print(f"原始模型大小：{os.path.getsize('model.pth')/1024:.1f} KB")
print(f"量化后模型大小：{os.path.getsize('quantized_unet.pth')/1024:.1f} KB")

# 量化验证模块
class QuantValidator:
    @staticmethod
    def validate_accuracy(orig_model, quant_model, test_loader):
        orig_model.eval()
        quant_model.eval()
        total_similarity = 0.0
        
        with torch.no_grad():
            for data in test_loader:
                orig_out = orig_model(data)
                quant_out = quant_model(data)
                similarity = torch.cosine_similarity(orig_out.flatten(), quant_out.flatten(), dim=0)
                total_similarity += similarity.item()
        
        avg_similarity = total_similarity / len(test_loader)
        logging.info(f"量化模型精度保持率: {avg_similarity:.2%}")
        return avg_similarity

# 精度验证
with torch.no_grad():
    test_input = torch.randn(1, 3, 256, 256)
    orig_output = model(test_input)
    quant_output = quantized_model(test_input)
    similarity = torch.cosine_similarity(orig_output.flatten(), quant_output.flatten(), dim=0)
    print(f"输出余弦相似度：{similarity.item():.4f}")


