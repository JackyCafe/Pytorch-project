import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():


    # 設定圖片路徑
    image_path = "datasets/animal/cat/cat001.jpg"

    # 1️⃣ 圖片預處理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 調整大小
        transforms.ToTensor(),  # 轉換為 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 標準化
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224) 加上 batch 維度

    # 2️⃣ 載入 CNN 模型 (ResNet18)
    model = models.resnet18(pretrained=True)
    model.eval()  # 設定為推論模式

    # 3️⃣ 設定 Hook 擷取指定層的特徵圖
    activation = {}

    def hook_fn(module, input, output):
        activation["layer_output"] = output.detach()  # 紀錄特徵圖

    # 選擇要查看的層
    layer_name = "layer4"  # 可以改成 "layer1"
    model.layer4[1].conv2.register_forward_hook(hook_fn)  # 設定 hook 到 layer4 的最後一個 conv

    # 4️⃣ 前向傳播
    with torch.no_grad():
        model(input_tensor)

    # 5️⃣ 取得特徵圖並視覺化
    feature_maps = activation["layer_output"].squeeze(0)  # 移除 batch 維度
    num_feature_maps = feature_maps.shape[0]  # 通道數

    # 設定畫布大小
    cols = 8
    rows = num_feature_maps // cols + (num_feature_maps % cols > 0)  # 自動計算行數
    fig, axes = plt.subplots(rows, cols, figsize=(7, 7))

    # 可視化每個特徵圖
    for i, ax in enumerate(axes.flat):
        if i < num_feature_maps:
            img = feature_maps[i].cpu().numpy()
            img = (img - np.min(img)) / (np.max(img) - np.min(img))  # 正規化 0-1
            ax.imshow(img, cmap='gray')
            ax.axis("off")
    plt.show()


if __name__ == '__main__':
    main()