import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def load_and_preprocess_data():
    """
    加载并预处理 MNIST 数据集。

    返回：
    - x_test: 归一化后的测试数据
    - y_test: one-hot 编码后的测试标签
    """
    # 加载 MNIST 数据集
    (_, _), (x_test, y_test) = mnist.load_data()  # 修改解包方式，忽略训练数据

    # 数据归一化
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0

    # 标签进行 one-hot 编码
    y_test = to_categorical(y_test, 10)

    return x_test, y_test


def load_and_evaluate_model(model_path, x_test, y_test):
    """
    加载训练好的模型并进行评估。

    参数：
    - model_path: 模型文件路径
    - x_test: 测试数据
    - y_test: 测试标签

    返回：
    - test_loss: 测试损失
    - test_accuracy: 测试准确率
    """
    # 加载训练好的模型
    model = load_model(model_path)

    # 评估模型
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    return test_loss, test_accuracy, model


def visualize_predictions(x_test, y_test, model):
    """
    可视化前6张测试图像及其预测结果。

    参数：
    - x_test: 测试数据
    - y_test: 测试标签
    - model: 已加载的训练模型
    """
    # 预测前6张图像的标签
    predictions = model.predict(x_test[:6])

    # 设置图像显示
    plt.figure(figsize=(10, 5))

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(x_test[i].squeeze(), cmap='gray')
        plt.title(f"True: {np.argmax(y_test[i])}, Pred: {np.argmax(predictions[i])}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # 1. 加载和预处理数据
    x_test, y_test = load_and_preprocess_data()

    # 2. 加载并评估模型
    test_loss, test_accuracy, model = load_and_evaluate_model('vggnet_mnist_model.h5', x_test, y_test)

    # 3. 打印测试结果
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")

    # 4. 可视化前6张图像及预测结果
    visualize_predictions(x_test, y_test, model)


if __name__ == "__main__":
    main()
