import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from vggnet import build_vgg


def load_and_preprocess_data():
    """
    加载并预处理 MNIST 数据集。

    返回：
    - x_train: 归一化后的训练数据
    - y_train: one-hot 编码后的训练标签
    - x_test: 归一化后的测试数据
    - y_test: one-hot 编码后的测试标签
    """
    # 加载 MNIST 数据集
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 数据归一化
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0

    # 标签进行 one-hot 编码
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def build_and_compile_model(num_classes=10):
    """
    构建并编译 VGGNet 模型。

    参数：
    - num_classes: 分类任务的类别数量，默认 10（MNIST）

    返回：
    - 编译好的模型
    """
    model = build_vgg(num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, epochs=5, batch_size=64):
    """
    训练并评估模型。

    参数：
    - model: 编译好的模型
    - x_train: 训练数据
    - y_train: 训练标签
    - x_test: 测试数据
    - y_test: 测试标签
    - epochs: 训练的周期数，默认为 5
    - batch_size: 每次训练批次大小，默认为 64

    返回：
    - history: 训练历史
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test),
                        callbacks=[early_stopping])

    return history


def save_model(model, filepath='vggnet_mnist_model.h5'):
    """
    保存训练好的模型。

    参数：
    - model: 训练好的模型
    - filepath: 模型保存路径，默认为 'vggnet_mnist_model.h5'
    """
    model.save(filepath)
    print(f"Model saved to {filepath}")


def main():
    # 1. 加载和预处理数据
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # 2. 构建并编译模型
    model = build_and_compile_model(num_classes=10)

    # 3. 训练并评估模型
    train_and_evaluate_model(model, x_train, y_train, x_test, y_test, epochs=5, batch_size=64)

    # 4. 保存模型
    save_model(model)


if __name__ == "__main__":
    main()
