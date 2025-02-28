import os


def read():
    train_path= 'datasets/chess_pieces/test/labels'

    # 取得所有標籤檔案
    for filename in os.listdir(train_path):
        file_path = os.path.join(train_path, filename)

        with open(file_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                class_id = int(parts[0])  # 讀取 class_id
                new_class_id = 1 if class_id < 7 else 7  # 變更 class_id
                parts[0] = str(new_class_id)  # 替換新的 class_id
                new_lines.append(" ".join(parts))  # 組合回字串

        # 覆寫檔案
        with open(file_path, 'w') as f:
            f.write("\n".join(new_lines))



if __name__ == '__main__':
    read()