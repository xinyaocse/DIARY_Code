import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential
import random


test_lunci = 10


def one_hot_labels(Y, dim):
    b = np.zeros((len(Y), dim))
    b[np.arange(len(Y)), Y] = 1

    return b


def my_read_data_encode_mnist():
    kcenter_list = []
    uncertain_list = []
    random_list = []
    adfa_list = []
    adfa_k_list = []
    knockoff = []

    for i in range(1, 41):
        temp = np.load("./data_generate/victim_encoder_out/mnist/kcenter_victim_" + str(i) + ".npy", allow_pickle=True).reshape(11, 2000)[:test_lunci]
        kcenter_list.append(temp)
    for i in range(1, 401):
        temp = np.load("./data_generate/victim_encoder_out/mnist/mnist_random_" + str(i) + ".npy", allow_pickle=True).reshape(11, 2000)[
               :test_lunci]
        random_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./data_generate/victim_encoder_out/mnist/uncertainty_victim_" + str(i) + ".npy", allow_pickle=True).reshape(11, 2000)[:test_lunci]
        uncertain_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./data_generate/victim_encoder_out/mnist/dfal_victim_" + str(i) + ".npy", allow_pickle=True).reshape(11, 2000)[:test_lunci]
        adfa_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./data_generate/victim_encoder_out/mnist/dfal-k_victim_" + str(i) + ".npy", allow_pickle=True).reshape(11, 2000)[:test_lunci]
        adfa_k_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./data_generate/victim_encoder_out/mnist/knockoff_victim_" + str(i) + ".npy", allow_pickle=True).reshape(10, 2000)[:test_lunci]
        knockoff.append(temp)

    for j in range(0, 400):
        if j < 200:
            t = uncertain_list[j % 40][0]
        else:
            t = kcenter_list[j % 40][0]

        random_list[j][0] = t

    random.shuffle(kcenter_list)
    random.shuffle(uncertain_list)
    random.shuffle(adfa_k_list)
    random.shuffle(adfa_list)
    random.shuffle(knockoff)

    kcenter_list = np.array(kcenter_list)
    uncertain_list = np.array(uncertain_list)
    random_list = np.array(random_list)
    adfa_list = np.array(adfa_list)
    adfa_k_list = np.array(adfa_k_list)
    knockoff_list=np.array(knockoff)
    # trainx = np.concatenate([random_list[:30], uncertain_list[:30], kcenter_list[:30], adfa_list[:30], adfa_k_list[:30]])  #
    malicious = np.concatenate([uncertain_list, kcenter_list, adfa_list, adfa_k_list,knockoff_list])  #
    benign = random_list

    return malicious, benign


# 双层LSTM模型
def DoubleLSTM(train_x, train_y):  # valid_x, valid_y,
    # 创建模型
    model = Sequential()
    model.add(LSTM(24, input_shape=(test_lunci, 2000), return_sequences=True))  # 返回所有节点的输出
    model.add(LSTM(12, return_sequences=False))  # 返回最后一个节点的输出
    model.add(Dense(2, activation='softmax'))
    # 查看网络结构
    model.summary()
    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # 训练模型
    model.fit(train_x, train_y, batch_size=6, epochs=10, verbose=2)  # , validation_data=(valid_x, valid_y)
    return model



if __name__ == "__main__":

    ##UNsupervise实验
    # train_x, train_y, test_x = my_read_data_mnist_shiyan()#原始
    malicious, benign = my_read_data_encode_mnist()

    train_x = np.concatenate([malicious[40:70], benign[:60]])#0-30 kcenter 40-70 uncertainty ,80-110 dfal , 120-150 dfalk
    y1 = [1] * 30
    y1=one_hot_labels(y1,2)
    y0 = [0] * 60
    y0=one_hot_labels(y0,2)
    train_y = np.concatenate([y1, y0])

    model = DoubleLSTM(train_x, train_y)

    out_bn = model.predict(benign[60:])
    label_bn = np.argmax(out_bn, axis=1)
    out_ml = model.predict(malicious)#np.concatenate([malicious[:80],malicious[120:]])
    label_ml = np.argmax(out_ml, axis=1)

    count_bn = 0
    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    c5 = 0

    time = 0
    for i in range(len(out_bn)):
        if label_bn[i] == 0:
            count_bn += 1
    for j in range(len(out_ml)):
        if j < 40 and label_ml[j] == 1:
            c1 += 1
        elif j < 80 and label_ml[j] == 1:
            c2 += 1
        elif j < 120 and label_ml[j] == 1:
            c3 += 1
        elif j < 160 and label_ml[j] == 1:
            c4 += 1
        elif j < 160 and label_ml[j] == 1:
            c4 += 1
    print("all_detect_length%s"% ((len(label_ml)+len(label_bn))))
    print("acc-benign:%s" % (count_bn/340))
    print("acc-kcenter:%s" % (c1/40))
    print("acc-uncertainty:%s" % (c2/40))
    print("acc-dfal:%s" % (c3/40))
    print("acc-dfalk:%s" % (c4 / 40))
    print("acc-knockoff:%s" % (c5 / 40))
