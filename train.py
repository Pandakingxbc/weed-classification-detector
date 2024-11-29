from ultralytics import YOLO

if __name__ == '__main__':
    # 选择合适的YOLO模型配置文件
    model_yaml = r"C:\Users\86137\Desktop\Computer Vision\ultralytics-main\ultralytics\cfg\models\11\yolo11.yaml"

    # 数据集配置文件路径
    data_yaml = r"C:\Users\86137\Desktop\Computer Vision\datasets\data.yaml"  # 你的数据配置文件

    # 如果有预训练模型，可以加载预训练权重；否则，可以从头开始训练
    pre_model = r"C:\Users\86137\Desktop\Computer Vision\yolo11l.pt"  # 预训练模型路径（可选）

    class_weights = {0: 1.0, 1: 3.0}  # 这里为 'mq' 类设置权重1.0，为 'weed' 类设置权重3.0

    # 初始化YOLO模型，指定任务为目标检测
    model = YOLO(model_yaml, task='detect').load(pre_model)

    # 开始训练
    results = model.train(
        data=data_yaml,  # 数据配置文件路径
        epochs=200,  # 训练轮数
        imgsz=640,  # 输入图像尺寸，建议选择 640x640
        batch=16,  # 每批次的图像数量，根据显存调整
        workers=4,  # 数据加载线程数量
        device='cuda',  # 使用的设备，'cuda' 表示使用 GPU，'cpu' 表示使用 CPU
        optimizer='Adam',  # 优化器选择（SGD, Adam等）
        lr0=0.001,  # 初始学习率
        lrf=0.01,  # 学习率衰减
        project='runs/train',  # 保存训练结果的路径
        name='yolo_train_1113',  # 训练结果保存的文件夹名称
        single_cls=False,  # 是否进行单类别检测（通常为False）
        cache='disk',  # 是否缓存数据（可以加快训练）
        amp=False,  # 禁用 AMP
        augment = True,

    )

    # 输出训练过程和结果
    print(results)
