import face_recognition
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class Face_detector(object):
	def cv2AddChineseText(self, img, text, position, textColor=(0, 255, 0), textSize=30):
		# 用于替代put_text的函数(可以在屏幕上显示中文)
		if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
			img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		# 创建一个可以在给定图像上绘图的对象
		draw = ImageDraw.Draw(img)
		# 字体的格式
		fontStyle = ImageFont.truetype(
			"simsun.ttc", textSize, encoding="utf-8")
		# 绘制文本
		draw.text(position, text, textColor, font=fontStyle)
		# 转换回OpenCV格式
		return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

	def face(self, path):
		# 存储指定路径下的人名列表
		known_names = []
		# 存储指定路径下每一张图片的特征
		known_encodings = []
		# print(os.listdir(path))

		for image_name in os.listdir(path):
			# os.listdir():  ['吕厅.png', '杨双菁.png', '翟世超.jpg']  得到path目录下包含所有文件名称的列表
			load_image = face_recognition.load_image_file(path + image_name)  # 加载图片 "path + image_name"得到文件夹下图片的完成路径
			# 此处获得的图片不是原始图片,而是三个通道交换后的结果
			image_face_encoding = face_recognition.face_encodings(load_image)[0]  # 获得128维特征值, shape:(128,)
			known_names.append(image_name.split(".")[0])
			known_encodings.append(image_face_encoding)

		# print(known_encodings)

		# 打开摄像头，0表示内置摄像头
		video_capture = cv2.VideoCapture(0)
		process_this_frame = True
		tracking_dict = {}
		while True:
			ret, frame = video_capture.read()
			# opencv的图像是BGR格式的，而我们需要是的RGB格式的，因此需要进行一个转换。
			rgb_frame = frame[:, :, ::-1]  # 将bgr转换为rgb
			# if process_this_frame:
			if ret:
				face_locations = face_recognition.face_locations(rgb_frame)  # 获得所有人脸位置
				# [(top, right, bottom, left),(),(),...()] 有几个人脸，list中的元素就有几个
				face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)  # 获得人脸特征值
				# [(128,),(128,),...,(128,)] 有几个人脸，list中就有几个元素
				face_names = []  # 存储出现在画面中人脸的名字
				for face_encoding in face_encodings:  # face_encodings不为空才会进入循环，即检测到人脸才会进入循环
					# 逐一去除检测到的人脸特征向量
					matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
					# 根据两个向量的内积计算相似度，根据阈值判断是否为同一个人脸
					# 返回值为布尔值列表，阈值越小，越严格
					if True in matches:
						# 当前检测到的人脸在存储的数据库中有对应值
						first_match_index = matches.index(True)
						print(first_match_index)
						name = known_names[first_match_index]
					# print("name", name)
					else:
						# 当前检测到的人脸没有在数据库中有对应值
						name = "unknown"
					face_names.append(name)  # 标记的人脸名字顺序与检测到的人脸顺序一致，二者相互对应
			else:
				print("检测中断")
				break
			# process_this_frame = not process_this_frame

			# for key in tracking_dict:
			# 	tracking_dict[key].append(-1)

			# 将捕捉到的人脸显示出来
			for (top, right, bottom, left), name in zip(face_locations, face_names):
				if (name in tracking_dict):
					tracking_dict[name].append((top, right, bottom, left))
				else:
					tracking_dict[name] = [(top, right, bottom, left)]

				if (len(tracking_dict[name]) > 15):
					tracking_dict[name].pop(0)
				for index in range(len(tracking_dict[name])):
					# if (tracking_dict[name][index]!=-1):

					temp_top, temp_right, temp_bottom, temp_left = tracking_dict[name][index]
					cv2.circle(frame, (
					int(temp_left + (temp_right - temp_left) / 2), int(temp_top + (temp_bottom - temp_top) / 2)), 10,
							   (0, 255, 0), -1)  # 画人脸矩形框
					if (index != 0):
						temp_top0, temp_right0, temp_bottom0, temp_left0 = tracking_dict[name][index - 1]
						cv2.line(frame, (int(temp_left0 + (temp_right0 - temp_left0) / 2),
										 int(temp_top0 + (temp_bottom0 - temp_top0) / 2)), (
								 int(temp_left + (temp_right - temp_left) / 2),
								 int(temp_top + (temp_bottom - temp_top) / 2)), (0, 255, 0), 5)

				cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  # 画人脸矩形框
				# 加上人名标签
				cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
				# font = cv2.FONT_HERSHEY_DUPLEX
				# cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
				frame = self.cv2AddChineseText(frame, name, (left + 30, bottom - 30), (0, 255, 0), 30)

			cv2.imshow('frame', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				print("停止检测")
				break

		video_capture.release()
		cv2.destroyAllWindows()

if __name__ == '__main__':
	face_dector = Face_detector()
	face_dector.face("./images/")  # 存放已知图像路径