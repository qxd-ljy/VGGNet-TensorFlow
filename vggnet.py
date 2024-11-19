from tensorflow.keras import layers, models


def build_vgg(input_shape=(28, 28, 1), num_classes=10):
    """
    构建 VGGNet 模型。

    参数：
    - input_shape: 输入数据的形状 (默认 (28, 28, 1)，适合处理 MNIST 数据)
    - num_classes: 分类任务的类别数（默认 10）

    返回：
    - model: 构建好的 VGGNet 模型
    """
    model = models.Sequential()

    # 第一层卷积块: 64个卷积核，大小3x3，ReLU激活
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # 池化层，2x2池化

    # 第二层卷积块: 128个卷积核，大小3x3，ReLU激活
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # 第三层卷积块: 256个卷积核，大小3x3，ReLU激活
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # 第四层卷积块: 512个卷积核，大小3x3，ReLU激活
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # 展平层，将3D输出转换为1D
    model.add(layers.Flatten())

    # 全连接层：两个具有4096个神经元的层，带有Dropout
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))  # 防止过拟合，50%神经元随机丢弃
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))

    # 输出层：根据类别数设置输出维度，使用softmax进行分类
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model
