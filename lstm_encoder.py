import numpy as np
from keras.layers import Dense
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Model
from keras.models import Sequential

test_lunci = 10


def one_hot_labels(Y, dim):
    b = np.zeros((len(Y), dim))
    b[np.arange(len(Y)), Y] = 1

    return b


def shufl(trainx):
    trainx = np.array(trainx)
    shuffle_ix = np.random.permutation(np.arange(len(trainx)))
    trainx = trainx[shuffle_ix]
    return trainx

def my_read_data_encode_flower():
    kcenter_list = []
    uncertain_list = []
    random_list = []
    adfa_list = []
    adfa_k_list = []
    random_t = []
    knockoff=[]
    for i in range(1, 41):
        temp = np.load("./out_flower/flower_kcenter" + str(i) + ".npy", allow_pickle=True).reshape(11, -1)[:test_lunci]
        kcenter_list.append(temp)
    for i in range(1, 401):
        temp = np.load("./out_flower/flower_random" + str(i) + ".npy", allow_pickle=True).reshape(11, -1)[:test_lunci]
        random_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./out_flower/flower_uncertainty" + str(i) + ".npy", allow_pickle=True).reshape(11, -1)[:test_lunci]
        uncertain_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./out_flower/flower_dfal" + str(i) + ".npy", allow_pickle=True).reshape(11, -1)[:test_lunci]
        adfa_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./out_flower/flower_adflk" + str(i) + ".npy", allow_pickle=True).reshape(11, -1)[:test_lunci]
        adfa_k_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./output_knockoff/flower/knockoff_victim_" + str(i) + ".npy", allow_pickle=True).reshape(10, -1)[:test_lunci]
        knockoff.append(temp)
    for j in range(0, 400):
        if j < 100:
            t = uncertain_list[j % 40][0]
        elif j < 200:
            t = kcenter_list[j % 40][0]
        elif j < 300:
            t = adfa_list[j % 40][0]
        else:
            t = adfa_k_list[j % 40][0]

        random_list[j][0] = t

    kcenter_list = np.array(kcenter_list)
    uncertain_list = np.array(uncertain_list)
    random_list = np.array(random_list)
    adfa_list = np.array(adfa_list)
    adfa_k_list = np.array(adfa_k_list)
    knockoff = np.array(knockoff)
    # trainx = np.concatenate([random_list[:30], uncertain_list[:30], kcenter_list[:30], adfa_list[:30], adfa_k_list[:30]])  #
    malicious = np.concatenate([kcenter_list, uncertain_list, adfa_list, adfa_k_list,knockoff])  #
    benign = random_list

    return malicious, benign

def my_read_data_encode_gtsr():
    kcenter_list = []
    uncertain_list = []
    random_list = []
    adfa_list = []
    adfa_k_list = []
    random_t = []
    knockoff=[]
    for i in range(1, 41):
        temp = np.load("./gtsr_victim/gtsr_kcenter_image" + str(i) + ".npy", allow_pickle=True).reshape(11, -1)[:test_lunci]
        kcenter_list.append(temp)
    for i in range(0, 400):
        temp = np.load("./gtsr_victim/gtsr_random_image" + str(i) + ".npy", allow_pickle=True).reshape(11, -1)[:test_lunci]
        random_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./gtsr_victim/gtsr_uncertainty_image" + str(i) + ".npy", allow_pickle=True).reshape(11, -1)[:test_lunci]
        uncertain_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./gtsr_victim/gtsr_adfl_image" + str(i) + ".npy", allow_pickle=True).reshape(11, -1)[:test_lunci]
        adfa_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./gtsr_victim/gtsr_adflk_image" + str(i) + ".npy", allow_pickle=True).reshape(11, -1)[:test_lunci]
        adfa_k_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./output_knockoff/gtsr/knockoff_victim_" + str(i) + ".npy", allow_pickle=True).reshape(10, -1)[:test_lunci]
        knockoff.append(temp)
    for j in range(0, 400):
        if j < 100:
            t = uncertain_list[j % 40][0]
        elif j < 200:
            t = kcenter_list[j % 40][0]
        elif j < 300:
            t = adfa_list[j % 40][0]
        else:
            t = adfa_k_list[j % 40][0]

        random_list[j][0] = t

    kcenter_list = np.array(kcenter_list)
    uncertain_list = np.array(uncertain_list)
    random_list = np.array(random_list)
    adfa_list = np.array(adfa_list)
    adfa_k_list = np.array(adfa_k_list)
    knockoff = np.array(knockoff)
    # trainx = np.concatenate([random_list[:30], uncertain_list[:30], kcenter_list[:30], adfa_list[:30], adfa_k_list[:30]])  #
    malicious = np.concatenate([kcenter_list, uncertain_list, adfa_list, adfa_k_list,knockoff])  #
    benign = random_list

    return malicious, benign

def my_read_data_encode_cifar():
    kcenter_list = []
    uncertain_list = []
    random_list = []
    adfa_list = []
    adfa_k_list = []
    random_t = []
    knockoff=[]
    for i in range(1, 41):
        temp = np.load("./out_cifar/kcet_c_" + str(i) + ".npy", allow_pickle=True).reshape(11, 2000)[:test_lunci]
        kcenter_list.append(temp)
    for i in range(0, 400):
        temp = np.load("./random_data/cifar_random_" + str(i) + ".npy", allow_pickle=True).reshape(11, 2000)[
               :test_lunci]
        random_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./out_cifar/unct_c_" + str(i) + ".npy", allow_pickle=True).reshape(11, 2000)[:test_lunci]
        uncertain_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./out_cifar/adfl_c_" + str(i) + ".npy", allow_pickle=True).reshape(11, 2000)[:test_lunci]
        adfa_list.append(temp)
    for i in range(0, 40):
        temp = np.load("./out_cifar/adfl-k_victim_" + str(i) + ".npy", allow_pickle=True).reshape(-1, 2000)[:test_lunci]
        adfa_k_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./output_knockoff/cifar/knockoff_victim_" + str(i) + ".npy", allow_pickle=True).reshape(10, -1)[:test_lunci]
        knockoff.append(temp)
    for j in range(0, 400):
        if j < 100:
            t = uncertain_list[j % 40][0]
        elif j < 200:
            t = kcenter_list[j % 40][0]
        elif j < 300:
            t = adfa_list[j % 40][0]
        else:
            t = adfa_k_list[j % 40][0]

        random_list[j][0] = t

    kcenter_list = np.array(kcenter_list)
    uncertain_list = np.array(uncertain_list)
    random_list = np.array(random_list)
    adfa_list = np.array(adfa_list)
    adfa_k_list = np.array(adfa_k_list)
    knockoff = np.array(knockoff)
    # trainx = np.concatenate([random_list[:30], uncertain_list[:30], kcenter_list[:30], adfa_list[:30], adfa_k_list[:30]])  #
    malicious = np.concatenate([kcenter_list,uncertain_list,  adfa_list, adfa_k_list,knockoff])  #
    benign = random_list

    return malicious, benign

def my_read_data_encode_mnist():
    kcenter_list = []
    uncertain_list = []
    random_list = []
    adfa_list = []
    adfa_k_list = []
    knockoff=[]
    for i in range(1, 41):
        temp = np.load("./out/kcenter_victim_" + str(i) + ".npy", allow_pickle=True).reshape(11, 2000)[:test_lunci]
        kcenter_list.append(temp)
    for i in range(0, 400):
        temp = np.load("./random_data/mnist_random_" + str(i) + ".npy", allow_pickle=True).reshape(11, 2000)[
               :test_lunci]
        random_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./out/uncertainty_victim_" + str(i) + ".npy", allow_pickle=True).reshape(11, 2000)[:test_lunci]
        uncertain_list.append(temp)
    for i in range(1, 41):
        temp = np.load("./out/adfa_victim_" + str(i) + ".npy", allow_pickle=True).reshape(11, 2000)[:test_lunci]
        adfa_list.append(temp)
    for i in range(0, 40):
        temp = np.load("./out/adfl-k_victim_" + str(i) + ".npy", allow_pickle=True).reshape(11, 2000)[:test_lunci]
        adfa_k_list.append(temp)
    for i in range(1, 41):
        # temp = np.load("./output_knockoff/mnist/knockoff_victim_" + str(i) + ".npy", allow_pickle=True).reshape(10, -1)[:test_lunci]
        temp = np.load("./mnist_test/random_victim_" + str(i) + ".npy", allow_pickle=True).reshape(-1, 2000)[:test_lunci]
        knockoff.append(temp)
    for j in range(0, 400):
        if j < 100:
            t = uncertain_list[j % 40][0]
        elif j < 200:
            t = kcenter_list[j % 40][0]
        elif j < 300:
            t = adfa_list[j % 40][0]
        else:
            t = adfa_k_list[j % 40][0]

        random_list[j][0] = t

    kcenter_list = np.array(kcenter_list)
    uncertain_list = np.array(uncertain_list)
    random_list = np.array(random_list)
    adfa_list = np.array(adfa_list)
    adfa_k_list = np.array(adfa_k_list)
    knockoff=np.array(knockoff)
    # trainx = np.concatenate([random_list[:30], uncertain_list[:30], kcenter_list[:30], adfa_list[:30], adfa_k_list[:30]])  #
    malicious = np.concatenate([kcenter_list,uncertain_list,  adfa_list, adfa_k_list,knockoff])  #
    benign = random_list

    return malicious, benign


# 双层LSTM模型
def DoubleLSTM(train_x, train_y, test_x, test_y):  # valid_x, valid_y,
    # 创建模型
    model = Sequential()
    model.add(LSTM(24, input_shape=(test_lunci, 2000), return_sequences=True))  # 返回所有节点的输出
    model.add(LSTM(12, return_sequences=False))  # 返回最后一个节点的输出
    # model.add(Dense(3, activation='softmax'))
    # 查看网络结构
    model.summary()
    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # 训练模型
    return model
    # model.fit(train_x, train_y, batch_size=6, epochs=10, verbose=2)#, validation_data=(valid_x, valid_y)
    #
    # # 评估模型
    # pre = model.evaluate(test_x, test_y, batch_size=500, verbose=2)
    # out=model.predict(test_x)
    # print('test_loss:', pre[0], '- test_acc:', pre[1])


if __name__ == "__main__":
    # mnist_datasets

    weidu=3400 #200*n,gtsr:8600,cifar,minst:2000,flower:3400

    malicious, benign = my_read_data_encode_flower_jdba()
    train_x = np.concatenate([malicious, benign[:200]])
    test_x = train_x

    train_x_all = np.concatenate([train_x, test_x])
    seq_out = train_x_all[:, 1:, :]
    visible = Input(shape=(test_lunci, weidu))
    encoder = LSTM(100, activation='relu')(visible)
    # define reconstruct decoder
    decoder1 = RepeatVector(test_lunci)(encoder)
    decoder1 = LSTM(100, activation='relu', return_sequences=True)(decoder1)
    decoder1 = TimeDistributed(Dense(weidu))(decoder1)
    # define predict decoder
    decoder2 = RepeatVector(test_lunci - 1)(encoder)
    decoder2 = LSTM(100, activation='relu', return_sequences=True)(decoder2)
    decoder2 = TimeDistributed(Dense(weidu))(decoder2)
    # tie it together
    model = Model(inputs=visible, outputs=[decoder1, decoder2])
    model.compile(optimizer='adam', loss='mse')
    # plot_model(model, show_shapes=True, to_file='composite_lstm_autoencoder.png')

    model.fit(train_x_all, [train_x_all, seq_out], epochs=100, verbose=0)
    # model=DoubleLSTM(train_x, train_y, test_x, test_y)#valid_x, valid_y,
    # model.fit(train_x, train_y, batch_size=6, epochs=10, verbose=2)
    # model.save("./model/encoder.h5")
    yhat = model.predict(malicious, verbose=2)
    tout = model.predict(benign, verbose=2)
    np.save("./lstm_encode_out/jdba_flower_lstm_encode_malicious_all_test.npy", yhat[0])
    np.save("./lstm_encode_out/jdba_flower_encode_benign_all_test.npy", tout[0])
    # out = model.predict(test_x)
    print(yhat)
