import os
import cv2

def load(path,detect_num,test_percentage,resize_h,resize_w):
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(0 , detect_num):
        images = os.listdir(path+'/'+str(i))
        image_num = len(images)
        test_num = int(image_num * (test_percentage/100.0))
        train_num = image_num - test_num

        for j in range(image_num):
            if images[j].find('.png') > 0 or images[j].find('.jpg') > 0 or images[j].find('.jpeg') > 0:
                image_path = path+'/'+str(i)+'/'+images[j]
                image = cv2.imread(image_path)
                image = cv2.resize(image,(resize_h,resize_w))

                if j < train_num:
                    x_train.append(image)
                    y_train.append(i)
                else:
                    x_test.append(image)
                    y_test.append(i)
    return x_train,x_test,y_train,y_test

