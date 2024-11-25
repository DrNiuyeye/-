import cv2
import numpy as np
import os
import csv
import pandas as pd
import glob
import matplotlib.pyplot as plt
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QPushButton, QLabel, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QRect
from PyQt5 import QtCore, QtGui, QtWidgets

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import load_model

@tf.keras.utils.register_keras_serializable()
class LeNet(tf.keras.Model):
    def __init__(self):
        super(LeNet, self).__init__()
        # 超参数设置
        self.conv1 = tf.keras.layers.Conv2D(6, (5, 5), activation='relu', padding='valid')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='valid')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(120, activation='relu')
        self.fc2 = tf.keras.layers.Dense(84, activation='relu')
        self.output_layer = tf.keras.layers.Dense(43, activation='softmax')

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        if training:
            x = tf.nn.dropout(x, rate=1 - 1)  # keep_prob为1时不应用dropout
        return self.output_layer(x)

    def get_config(self):
        config = super(LeNet, self).get_config()
        config.pop('name', None)  # 删除多余的 'name' 参数
        # 添加其他需要的参数
        return config


    @classmethod
    def from_config(cls, config):
        config.pop('trainable', None)  # 删除 'trainable' 参数
        config.pop('dtype', None)       # 删除 'dtype' 参数
        return cls(**config)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(949, 731)
        MainWindow.setStyleSheet("background-color: rgb(109, 165, 213);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QRect(100, 340, 93, 31))
        self.pushButton.setStyleSheet("background-color: rgb(85, 170, 255);")
        self.pushButton.setObjectName("pushButton")

        self.tableWidget = QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QRect(330, 550, 592, 144))
        self.tableWidget.horizontalHeader().setStyleSheet(
            "QHeaderView::section{background-color:rgb(155, 194, 230);font:11pt '宋体';color: black;};")
        self.tableWidget.setStyleSheet("background-color: rgb(210, 180, 140);")
        self.tableWidget.setLineWidth(1)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setRowCount(3)
        self.tableWidget.setVerticalHeaderItem(0, QTableWidgetItem())
        self.tableWidget.setVerticalHeaderItem(1, QTableWidgetItem())
        self.tableWidget.setVerticalHeaderItem(2, QTableWidgetItem())
        self.tableWidget.setHorizontalHeaderItem(0, QTableWidgetItem())
        self.tableWidget.setHorizontalHeaderItem(1, QTableWidgetItem())
        self.tableWidget.setHorizontalHeaderItem(2, QTableWidgetItem())
        self.tableWidget.horizontalHeader().setDefaultSectionSize(155)
        self.tableWidget.verticalHeader().setVisible(False)

        self.label = QLabel(self.centralwidget)
        self.label.setGeometry(QRect(230, 20, 451, 51))
        self.label.setObjectName("label")

        # 将 textBrowser 替换为 QLabel
        self.imageLabel = QLabel(self.centralwidget)
        self.imageLabel.setGeometry(QRect(330, 120, 590, 401))
        self.imageLabel.setObjectName("imageLabel")
        self.imageLabel.setScaledContents(True)  # 设置图片填充充满 QLabel 区域
        self.imageLabel.setStyleSheet("border: 3px solid black;")  # 设置黑色边框，宽度为3像素

        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setGeometry(QRect(30, 215, 35, 35))
        self.label_2.setTextFormat(QtCore.Qt.RichText)
        self.label_2.setObjectName("label_2")

        self.pushButton_2 = QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QRect(100, 220, 91, 28))
        self.pushButton_2.setStyleSheet("background-color: rgb(85, 170, 255);")
        self.pushButton_2.setObjectName("pushButton_2")

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tableWidget.setColumnWidth(0, 100)  # 设置首列宽度为100像素
        self.tableWidget.setColumnWidth(1, 490)
        self.pushButton.setText(_translate("MainWindow", "开始运行>"))
        self.tableWidget.verticalHeaderItem(0).setText(_translate("MainWindow", "新建行"))
        self.tableWidget.verticalHeaderItem(1).setText(_translate("MainWindow", "新建行"))
        self.tableWidget.verticalHeaderItem(2).setText(_translate("MainWindow", "新建行"))
        self.tableWidget.horizontalHeaderItem(0).setText(_translate("MainWindow", "序号"))
        self.tableWidget.horizontalHeaderItem(1).setText(_translate("MainWindow", "识别结果"))
        self.label.setText(_translate("MainWindow", "<html><body style=\" font-family:'SimSun'; font-size:26pt; font-weight:600; color:#000000;\">交通标志检测识别系统</body></html>"))
        pixmap = QPixmap("traffic_picture/document.png")
        self.label_2.setPixmap(pixmap.scaled(self.label_2.size(), aspectRatioMode=1))
        self.pushButton_2.setText(_translate("MainWindow", "选择图片"))

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        # 设置窗口标题
        self.setWindowTitle("交通标志检测识别系统")
        self.setupUi(self)
        self.image_path = ""  # 初始化空路径

        # 连接按钮的点击事件到槽函数
        self.pushButton_2.clicked.connect(self.choose_image)
        self.pushButton.clicked.connect(self.run_model)

    def choose_image(self):
        # 打开文件对话框，选择图片文件
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")

        # 如果用户选择了文件，将图片显示到 QLabel 中
        if file_path:
            self.image_path = file_path  # 更新 image_path 为选中的图片路径
            pixmap = QPixmap(file_path)  # 加载图片
            self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.size(), aspectRatioMode=1))  # 设置为 QLabel 的图片
            self.imageLabel.setAlignment(Qt.AlignCenter)  # 设置图片居中显示

            # 这里可以调用处理图像的函数
            self.process_image(self.image_path)
            self.Cal_message()

    def process_image(self, image_path):
        # 先删除之前留下的图片
        # 图片文件夹路径
        dele_folder = r"F:\\baidudownload\\GTSRB_Test_Images\\GTSRB\\Final_Test\\Images"

        # 遍历文件夹中的每张图片
        for filename in os.listdir(dele_folder):
            if filename.endswith(".ppm"):  # 确保只处理 .ppm 文件
                dele_path = os.path.join(dele_folder, filename)
                os.remove(dele_path)

        # 图像处理部分
        RGB = cv2.imread(image_path)

        # 将 BGR 转换为 RGB
        RGB = cv2.cvtColor(RGB, cv2.COLOR_BGR2RGB)

        # 将 RGB 转换到 LAB 颜色空间
        LAB = cv2.cvtColor(RGB, cv2.COLOR_RGB2Lab)

        # 提取亮度通道
        L = LAB[:, :, 0] / 255.0  # LAB 中的亮度通道

        # 应用 CLAHE
        clahe = cv2.createCLAHE(clipLimit=0.005, tileGridSize=(8, 8))
        L_clahe = clahe.apply((L * 255).astype(np.uint8))

        # 将增强后的亮度通道放回 LAB 颜色空间
        LAB[:, :, 0] = L_clahe

        # 转换回 RGB 颜色空间
        J = cv2.cvtColor(LAB, cv2.COLOR_Lab2RGB)

        # 转换到 HSV 颜色空间
        hsv = cv2.cvtColor(J, cv2.COLOR_RGB2HSV)

        # 提取 H、S 和 V 分量
        h1 = hsv[:,:,0] / 180.0  # OpenCV 的 H 分量范围是 [0, 179]
        s1 = hsv[:,:,1] / 255.0  # S 分量范围是 [0, 255]
        v1 = hsv[:,:,2] / 255.0  # V 分量范围是 [0, 255]

        # 提取红色分量
        hsvR = ((h1 <= 0.056) | (h1 >= 0.740)) & (s1 >= 0.169) & (s1 <= 1.0) & (v1 >= 0.180) & (v1 <= 1.0)

        # 转换为 uint8 格式以便进行轮廓检测
        hsvR_uint8 = (hsvR * 255).astype(np.uint8)

        # 寻找轮廓
        contours, _ = cv2.findContours(hsvR_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 初始化参数
        area_threshold = 800  # 面积阈值，可以根据需要调整

        # 在原图上绘制边框
        output_image = RGB.copy()
        save_counter = 0  # 用于保存切割图片时命名

        for contour in contours:
            area = cv2.contourArea(contour)  # 计算轮廓面积
            if area > area_threshold:  # 只考虑大于阈值的轮廓
                # 计算轮廓的离心率
                (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
                if MA < ma:
                    MA, ma = ma, MA  # 交换 MA 和 ma
                if MA > 0 and ma > 0:  # 确保 MA 和 ma 都大于零
                    eccentricity = np.sqrt(1 - (ma / MA) ** 2)  # 离心率计算
                    print(f"离心率: {eccentricity}")

                    # 如果离心率符合条件，则绘制边框
                    if eccentricity > 0 and eccentricity < 0.5 :
                        # 计算边框坐标
                        x, y, w, h = cv2.boundingRect(contour)
                        padding = 6
                        x -= padding
                        y -= padding
                        w += 2 * padding
                        h += 2 * padding
                        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

                        # 切割并保存图片
                        cropped_image = RGB[y:y + h, x:x + w]
                        crop_image_path = f"F:\\baidudownload\\GTSRB_Test_Images\\GTSRB\\Final_Test\\Images\\0000{save_counter}.ppm"  # 保存的图片文件名
                        cv2.imwrite(crop_image_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))  # 保存为PPM格式
                        save_counter += 1
                        # 显示结果图像
                        # 将处理后的图像转换为 QPixmap 格式
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)  # 转换为 BGR 格式
        height, width, channel = output_image.shape
        bytes_per_line = 3 * width
        q_image = QtGui.QImage(output_image.data, width, height, bytes_per_line, QtGui.QImage.Format_BGR888)
        pixmap = QPixmap(q_image)

        # 设置 QLabel 显示处理后的图像
        self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.size(), QtCore.Qt.KeepAspectRatio))

    def Cal_message(self):
        # 图片文件夹路径
        image_folder = r"F:\\baidudownload\\GTSRB_Test_Images\\GTSRB\\Final_Test\\Images"

        # 要写入的 CSV 文件名
        csv_file = os.path.join(image_folder, "GT-final_test.csv")

        # 读取原始文件内容，保留第一行
        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=';')
            rows = list(reader)
        # 获取第一行
        header = rows[0] if rows else []
        # 以写入模式打开文件，然后使用 truncate() 清空内容
        with open(csv_file, mode='r+', encoding='utf-8') as file:
            file.truncate(0)


        # 遍历文件夹中的每张图片
        tt = 0
        for filename in os.listdir(image_folder):
            if filename.endswith(".ppm"):  # 确保只处理 .ppm 文件
                image_path = os.path.join(image_folder, filename)

                # 读取图像
                RGB = cv2.imread(image_path)
                
                # 检查图像是否成功读取
                if RGB is None:
                    print(f"无法读取图像 {filename}")
                    continue
                
                # 获取图像的高度和宽度
                height, width = RGB.shape[:2]

                # 将 BGR 转换为 RGB
                RGB = cv2.cvtColor(RGB, cv2.COLOR_BGR2RGB)

                # 将 RGB 转换到 LAB 颜色空间
                LAB = cv2.cvtColor(RGB, cv2.COLOR_RGB2Lab)

                # 提取亮度通道
                L = LAB[:, :, 0] / 255.0

                # 应用 CLAHE
                clahe = cv2.createCLAHE(clipLimit=0.005, tileGridSize=(8, 8))
                L_clahe = clahe.apply((L * 255).astype(np.uint8))

                # 将增强后的亮度通道放回 LAB 颜色空间
                LAB[:, :, 0] = L_clahe

                # 转换回 RGB 颜色空间
                J = cv2.cvtColor(LAB, cv2.COLOR_Lab2RGB)

                # 转换到 HSV 颜色空间
                hsv = cv2.cvtColor(J, cv2.COLOR_RGB2HSV)

                # 提取 H、S 和 V 分量
                h1 = hsv[:, :, 0] / 180.0
                s1 = hsv[:, :, 1] / 255.0
                v1 = hsv[:, :, 2] / 255.0

                # 提取红色分量
                hsvR = ((h1 <= 0.056) | (h1 >= 0.740)) & (s1 >= 0.169) & (s1 <= 1.0) & (v1 >= 0.180) & (v1 <= 1.0)

                # 转换为 uint8 格式以便进行轮廓检测
                hsvR_uint8 = (hsvR * 255).astype(np.uint8)

                # 寻找轮廓
                contours, _ = cv2.findContours(hsvR_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 初始化参数
                area_threshold = 800  # 面积阈值

                # 在原图上绘制边框
                output_image = RGB.copy()

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > area_threshold:
                        # 计算轮廓的离心率
                        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
                        if MA < ma:
                            MA, ma = ma, MA
                        if MA > 0 and ma > 0:
                            eccentricity = np.sqrt(1 - (ma / MA) ** 2)
                            print(f"{filename} - 离心率: {eccentricity}")

                            # 如果离心率符合条件，则绘制边框并保存数据
                            if 0 < eccentricity < 0.5:
                                # 计算边框坐标
                                x, y, w, h = cv2.boundingRect(contour)
                                
                                padding = 1
                                x -= padding
                                y -= padding
                                w += 2 * padding
                                h += 2 * padding

                                # 写入 CSV 文件
                                data_row = [filename, width, height, x, y, x + w, y + h]
                                with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                                    writer = csv.writer(file, delimiter=';')
                                    if tt == 0:
                                        writer.writerow(header)
                                    writer.writerow(data_row)
            tt += 1
            

    
    def run_model(self):
        def load_image(image_file):
            return plt.imread(image_file) # 读取图片

        # 读取图片并调整图片尺寸为(32,32)
        def resize_image(image_file, shape=(32,32)): # image_file:图片文件地址
            image_list=[]
            for image_file_n in image_file:
                image_file_n
                image=load_image(image_file_n)
                image=cv2.resize(image, shape)
                image_list.append(image)
            image=np.array(image_list)
            return image  # 输出:存储的调整后的图片文件


        sign_name_df = pd.read_csv("E:\\python\\.vscode\\GTSRB_Training_Images\\GTSRB\\signnames.csv", index_col='ClassId') # 读取各交通标志名并存储
        SIGN_NAMES = sign_name_df.SignName.values  # 获取交通标志名称，存储在SIGN_NAMES

        Test_IMAGE_DIR = "F:\\baidudownload\\GTSRB_Test_Images\\GTSRB\\Final_Test" # 此处为文件夹地址

        test_file=glob.glob(os.path.join(Test_IMAGE_DIR, '*/GT-*.csv'))

        folder = test_file[0].split('\\')[5]

        test_df = pd.read_csv(test_file[0], sep=';')  # 读取文件夹中csv文件
        test_df['Filename'] = test_df['Filename'].apply(lambda x: os.path.join(Test_IMAGE_DIR,folder, x)) # 将‘Filename'列内容延展，增加文件地址

        X_test = resize_image(test_df['Filename'].values) #读取图片并调整尺寸
        X_test = X_test.astype('float32')

        # 加载模型
        model = load_model("E:\\python\\.vscode\\Traffic_Signal_Classifier-CNN\\cnn_model\\lenet.keras")

        # 获取测试集的预测结果
        predictions = model.predict(X_test)

        # 如果需要，可以将预测结果转换为类标签
        predicted_classes = predictions.argmax(axis=1)
        predicted_labels = [SIGN_NAMES[class_id] for class_id in predicted_classes]
        data = []
        rank = 1
        for label in predicted_labels:
            data.append([str(rank),label])
            rank += 1

        # 清空 tableWidget 中的所有数据
        self.tableWidget.clearContents()

        # 将数据写入tableWidget
        for row, rowData in enumerate(data):
            for column, value in enumerate(rowData):
                # 创建QTableWidgetItem对象
                item = QTableWidgetItem(value)
                # 将item写入表格指定的单元格
                self.tableWidget.setItem(row, column, item)

    



if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())
