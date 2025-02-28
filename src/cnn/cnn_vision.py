import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch

def hook_fn(module, input, output):
    activation["layer1.0.conv2"] = output.detach()  # 紀錄第一層的輸出結果


def main():
    image_path = "./datasets/animal/cat/cat001.jpg"
    # 1. **圖片預處理**
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 調整大小
        transforms.ToTensor(),  # 轉換為 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 標準化
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # 增加 batch 維度 (1,3,224,224)
    model = models.resnet18(pretrained=True)
    model.eval()  # 設定為推論模式
    print(model)
    # 註冊 hook 在 ResNet 的第一層 Conv2D
    model.layer1[1].conv2.register_forward_hook(hook_fn)
    # 進行前向傳播
    with torch.no_grad():
        model(input_tensor)
    feature_maps = activation["layer1.0.conv2"].squeeze(0)  # (64, 112, 112) -> 64 個特徵圖
    num_feature_maps = feature_maps.shape[0]  # ResNet18 的 conv1 輸出 64 個通道

    cols = 8
    rows = num_feature_maps // cols  # 每列 8 張圖
    fig, axes = plt.subplots(rows, cols, figsize=(32, 32))

    for i, ax in enumerate(axes.flat):
        if i < num_feature_maps:
            img = feature_maps[i].cpu().numpy()  # 轉換為 numpy
            ax.imshow(img, cmap='gray')  # 顯示特徵圖
            ax.axis("off")  # 移除座標軸
    plt.show()
    plt.imshow(image)
    plt.show()




if __name__ == '__main__':
    activation = {}  # 用來儲存輸出特徵圖
    main()
