import os

import numpy as np
import torch

from utils import l2_normalize


def get_normal_vector(model, train_normal_loader_for_test, cal_vec_batch_size, feature_dim, use_cuda):
    total_batch = int(len(train_normal_loader_for_test))
    print("=====================================Calculating Average Normal Vector=====================================")
    if use_cuda:
        normal_vec = torch.zeros((1, 512)).cuda()
    else:
        normal_vec = torch.zeros((1, 512))
    for batch, (normal_data, idx) in enumerate(train_normal_loader_for_test):
        if use_cuda:
            normal_data = normal_data.cuda()
        _, outputs = model(normal_data)
        outputs = outputs.detach()
        normal_vec = (torch.sum(outputs, dim=0) + normal_vec * batch * cal_vec_batch_size) / (
                (batch + 1) * cal_vec_batch_size)
        print(f'Calculating Average Normal Vector: Batch {batch + 1} / {total_batch}')
    normal_vec = l2_normalize(normal_vec)
    return normal_vec


def acc_count(acc_m, acc_n):
    return 0.6 * acc_m + 0.4 * acc_n


def printout(th, acc_1, acc_2, acc_3, acc_4, acc_5, acc_n):
    print("th=" + str(th))
    th = int(th * 100)

    print(
        "kcenter_acc:%.3f,uncertainty_acc:%.3f,adfl_acc:%.3f,adflk_acc:%.3f ,knockoff_acc:%.3f ,de_acc_1:%.3f ,de_acc_2:%.3f ,de_acc_3:%.3f ,de_acc_4:%.3f ,de_acc_5:%.3f, bn_acc:%.3f" % (
            acc_1[th], acc_2[th], acc_3[th], acc_4[th], acc_5[th], acc_count(acc_1[th], acc_n[th]),
            acc_count(acc_2[th], acc_n[th]), acc_count(acc_3[th], acc_n[th]), acc_count(acc_4[th], acc_n[th]),
            acc_count(acc_5[th], acc_n[th]),
            acc_n[th]))


def print_allacc(acc_n,acc_a):
    print("th=0.95 all_acc:%.3f" % ((acc_n[95]*0.4+0.6*acc_a[95])))
    print("th=0.90 all_acc:%.3f" % ((acc_n[90]*0.4+0.6*acc_a[90])))
    print("th=0.85 all_acc:%.3f" % ((acc_n[85]*0.4+0.6*acc_a[85])))
    print("th=0.80 all_acc:%.3f" % ((acc_n[80]*0.4+0.6*acc_a[80])))
    print("th=0.75 all_acc:%.3f" % ((acc_n[75]*0.4+0.6*acc_a[75])))
    print("th=0.70 all_acc:%.3f" % ((acc_n[70]*0.4+0.6*acc_a[70])))
    print("th=0.65 all_acc:%.3f" % ((acc_n[65]*0.4+0.6*acc_a[65])))
    print("th=0.60 all_acc:%.3f" % ((acc_n[60]*0.4+0.6*acc_a[60])))
    print("th=0.55 all_acc:%.3f" % ((acc_n[55]*0.4+0.6*acc_a[55])))
    print("th=0.50 all_acc:%.3f" % ((acc_n[50]*0.4+0.6*acc_a[50])))
    print("th=0.45 all_acc:%.3f" % ((acc_n[45]*0.4+0.6*acc_a[45])))
    print("th=0.40 all_acc:%.3f" % ((acc_n[40]*0.4+0.6*acc_a[40])))
    print("th=0.35 all_acc:%.3f" % ((acc_n[35]*0.4+0.6*acc_a[35])))


def split_acc_diff_threshold(model, normal_vec, use_cuda):
    """
    Search the threshold that split the scores the best and calculate the corresponding accuracy
    """

    legth=10
    print("================================================Evaluating================================================")
    data_bn = np.load("../lstm_encode_out/mnist_lstm_encode_benign_all.npy",
                      allow_pickle=True) # cifar train:normal_150 malicious:120
    data_ml = np.load("../lstm_encode_out/mnist_lstm_encode_malicious_all.npy", allow_pickle=True)
    data_ml = data_ml[:, :legth, :]
    data_bn = data_bn[:, :legth, :]
    bn_test = data_bn[200:]
    ml_test = np.concatenate([data_ml[30:40], data_ml[70:80], data_ml[110:120], data_ml[150:160], data_ml[190:]])

    total_n = 0
    total_a = 0
    threshold = np.arange(0., 1., 0.01)
    total_correct_a = np.zeros(threshold.shape[0])
    total_correct_n = np.zeros(threshold.shape[0])
    total_correct_kcenter = np.zeros(threshold.shape[0])
    total_correct_uncertainty = np.zeros(threshold.shape[0])
    total_correct_adfl = np.zeros(threshold.shape[0])
    total_correct_adflk = np.zeros(threshold.shape[0])
    total_correct_knockoff = np.zeros(threshold.shape[0])
    nun_class = 10
    sim_k = 0
    sim_u = 0
    sim_a = 0
    sim_ak = 0
    sim_r = 0
    sim_knockoff = 0
    for i in range(len(bn_test) // 10):
        source = torch.from_numpy(bn_test[i * 10:(i + 1) * 10]).cuda()
        label = torch.zeros(10).cuda()
        total_n += source.size(0)
        _, outputs = model(torch.reshape(torch.unsqueeze(source, dim=1), (10, 1, legth, 200, -1)))
        similarity_n = torch.mm(outputs, normal_vec.t())
        t_sim = similarity_n.to('cpu')
        sim_r = np.mean(t_sim.detach().numpy())
        for n in range(len(threshold)):
            prediction = (similarity_n >= threshold[n])
            count = prediction.squeeze().sum().item()
            total_correct_n[n] += count

        # print(f'Evaluating normal: Batch {i + 1} / {1}')
        # print('\n')
    for j in range(len(ml_test) // 10):
        source = torch.from_numpy(ml_test[j * 10:(j + 1) * 10]).cuda()
        # label = torch.ones(10).cuda()
        total_a += source.size(0)
        _, outputs = model(torch.reshape(torch.unsqueeze(source, dim=1), (10, 1, legth, 200, -1)))
        similarity = torch.mm(outputs, normal_vec.t())
        t_similarity = similarity.to('cpu')
        if j == 0:
            sim_k = np.mean(t_similarity.detach().numpy())
        elif j == 1:
            sim_u = np.mean(t_similarity.detach().numpy())
        elif j == 2:
            sim_a = np.mean(t_similarity.detach().numpy())
        elif j == 3:
            sim_ak = np.mean(t_similarity.detach().numpy())
        elif j == 4:
            sim_knockoff = np.mean(t_similarity.detach().numpy())

        for o in range(len(threshold)):
            prediction = (similarity <= threshold[o])
            count = prediction.squeeze().sum().item()
            total_correct_a[o] += count
            if j == 0:
                total_correct_kcenter[o] += count
            elif j == 1:
                total_correct_uncertainty[o] += count
            elif j == 2:
                total_correct_adfl[o] += count
            elif j == 3:
                total_correct_adflk[o] += count
            elif j == 4:
                total_correct_knockoff[o] += count

    # acc
    acc_kcenter = [(correct_n / nun_class) for correct_n in total_correct_kcenter]
    acc_uncertain = [(correct_n / nun_class) for correct_n in total_correct_uncertainty]
    acc_adfl = [(correct_n / nun_class) for correct_n in total_correct_adfl]
    acc_adflk = [(correct_n / nun_class) for correct_n in total_correct_adflk]
    acc_knockoff = [(correct_n / nun_class) for correct_n in total_correct_knockoff]

    acc_n = [(correct_n / total_n) for correct_n in total_correct_n]
    acc_a = [(correct_a / total_a) for correct_a in total_correct_a]
    acc = [((total_correct_n[i] + total_correct_a[i]) / (total_n + total_a)) for i in range(len(threshold))]
    best_acc = np.max(acc)
    idx = np.argmax(acc)

    print("th=best.kcenter_acc:%f,uncertainty_acc；%f,adfl_acc:%f,adflk_acc:%f,bn_acc=%f " % (
        acc_kcenter[idx], acc_uncertain[idx], acc_adfl[idx],
        acc_adflk[idx], acc_n[idx]))
    printout(0.95, acc_kcenter, acc_uncertain, acc_adfl, acc_adflk, acc_knockoff, acc_n)
    printout(0.9, acc_kcenter, acc_uncertain, acc_adfl, acc_adflk, acc_knockoff, acc_n)
    printout(0.8, acc_kcenter, acc_uncertain, acc_adfl, acc_adflk, acc_knockoff, acc_n)
    printout(0.7, acc_kcenter, acc_uncertain, acc_adfl, acc_adflk, acc_knockoff, acc_n)
    printout(0.6, acc_kcenter, acc_uncertain, acc_adfl, acc_adflk, acc_knockoff, acc_n)
    printout(0.5, acc_kcenter, acc_uncertain, acc_adfl, acc_adflk, acc_knockoff, acc_n)
    printout(0.4, acc_kcenter, acc_uncertain, acc_adfl, acc_adflk, acc_knockoff, acc_n)
    printout(0.3, acc_kcenter, acc_uncertain, acc_adfl, acc_adflk, acc_knockoff, acc_n)
    printout(0.2, acc_kcenter, acc_uncertain, acc_adfl, acc_adflk, acc_knockoff, acc_n)
    print_allacc(acc_n,acc_a)
    # print("th=0.95.kcenter_acc:%f,uncertainty_acc；%f,adfl_acc:%f,adflk_acc:%f all_acc:%f,bn_acc=%f" % (
    #     acc_kcenter[95], acc_uncertain[95], acc_adfl[95], acc_adflk[95], acc[95], acc_n[95]))
    # print("th=0.9.kcenter_acc:%f,uncertainty_acc；%f,adfl_acc:%f,adflk_acc:%f all_acc:%f,bn_acc=%f" % (
    #     acc_kcenter[90], acc_uncertain[90], acc_adfl[90], acc_adflk[90], acc[90], acc_n[90]))
    # print("th=0.85.kcenter_acc:%f,uncertainty_acc；%f,adfl_acc:%f,adflk_acc:%f all_acc:%f,bn_acc=%f" % (
    #     acc_kcenter[85], acc_uncertain[85], acc_adfl[85], acc_adflk[85], acc[85], acc_n[85]))
    # print("th=0.8.kcenter_acc:%f,uncertainty_acc；%f,adfl_acc:%f,adflk_acc:%f all_acc:%f,bn_acc=%f" % (
    #     acc_kcenter[80], acc_uncertain[80], acc_adfl[80], acc_adflk[80], acc[80], acc_n[80]))
    # print("th=0.7.kcenter_acc:%f,uncertainty_acc；%f,adfl_acc:%f,adflk_acc:%f all_acc:%f,bn_acc=%f" % (
    #     acc_kcenter[80], acc_uncertain[80], acc_adfl[80], acc_adflk[80], acc[80], acc_n[70]))
    # print("th=0.65.kcenter_acc:%f,uncertainty_acc；%f,adfl_acc:%f,adflk_acc:%f all_acc:%f,bn_acc=%f" % (
    #     acc_kcenter[80], acc_uncertain[80], acc_adfl[80], acc_adflk[80], acc[80], acc_n[60]))
    # print("th=0.6.kcenter_acc:%f,uncertainty_acc；%f,adfl_acc:%f,adflk_acc:%f all_acc:%f,bn_acc=%f" % (
    #     acc_kcenter[80], acc_uncertain[80], acc_adfl[80], acc_adflk[80], acc[80], acc_n[50]))
    # print("similarity:random:%f,kcenter_acc:%f,uncertainty_acc；%f,adfl_acc:%f,adflk_acc:%f" % (
    #     sim_r, sim_k, sim_u, sim_a, sim_ak))

    best_threshold = idx * 0.01

    return best_acc, best_threshold, acc_n[idx], acc_a[idx], acc, acc_n, acc_a


def split_acc_diff_threshold_singleattack(model, normal_vec,  use_cuda):
    """
    Search the threshold that split the scores the best and calculate the corresponding accuracy
    """

    print("================================================Evaluating================================================")
    data_bn = np.load("../lstm_encode_out/mnist_lstm_encode_benign_all.npy",
                      allow_pickle=True)  # cifar train:normal_150 malicious:120
    data_ml = np.load("../lstm_encode_out/mnist_lstm_encode_malicious_all.npy", allow_pickle=True)
    bn_test = data_bn[200:]
    # ml_test=np.concatenate([data_ml[:80],data_ml[120:]])
    ml_test = data_ml

    total_n = 0
    total_a = 0
    threshold = np.arange(0., 1., 0.01)
    total_correct_a = np.zeros(threshold.shape[0])
    total_correct_n = np.zeros(threshold.shape[0])
    total_correct_1 = np.zeros(threshold.shape[0])
    total_correct_2 = np.zeros(threshold.shape[0])
    total_correct_3 = np.zeros(threshold.shape[0])
    total_correct_4 = np.zeros(threshold.shape[0])
    total_correct_5 = np.zeros(threshold.shape[0])
    nun_class = 40
    sim_1 = 0
    sim_2 = 0
    sim_3 = 0
    sim_4 = 0
    sim_5 = 0
    sim_r = 0
    for i in range(len(bn_test) // 10):
        source = torch.from_numpy(bn_test[i * 10:(i + 1) * 10]).cuda()
        label = torch.zeros(10).cuda()
        total_n += source.size(0)
        _, outputs = model(torch.reshape(torch.unsqueeze(source, dim=1), (10, 1, 10, 200, -1)))
        similarity_n = torch.mm(outputs, normal_vec.t())
        t_sim = similarity_n.to('cpu')
        sim_r = np.mean(t_sim.detach().numpy())
        for n in range(len(threshold)):
            prediction = (similarity_n >= threshold[n])
            count = prediction.squeeze().sum().item()
            total_correct_n[n] += count

        # print(f'Evaluating normal: Batch {i + 1} / {1}')
        # print('\n')
    for j in range(len(ml_test) // 10):
        source = torch.from_numpy(ml_test[j * 10:(j + 1) * 10]).cuda()
        # label = torch.ones(10).cuda()
        total_a += source.size(0)
        _, outputs = model(torch.reshape(torch.unsqueeze(source, dim=1), (10, 1, 10, 200, -1)))
        similarity = torch.mm(outputs, normal_vec.t())
        t_similarity = similarity.to('cpu')
        if j < 4:
            sim_1 += np.mean(t_similarity.detach().numpy())
        elif j < 8:
            sim_2 += np.mean(t_similarity.detach().numpy())
        elif j < 12:
            sim_3 += np.mean(t_similarity.detach().numpy())
        elif j < 16:
            sim_4 += np.mean(t_similarity.detach().numpy())
        elif j < 20:
            sim_5 += np.mean(t_similarity.detach().numpy())

        for o in range(len(threshold)):
            prediction = (similarity <= threshold[o])
            count = prediction.squeeze().sum().item()
            total_correct_a[o] += count
            if j < 4:
                total_correct_1[o] += count
            elif j < 8:
                total_correct_2[o] += count
            elif j < 12:
                total_correct_3[o] += count
            elif j < 16:
                total_correct_4[o] += count
            elif j < 20:
                total_correct_5[o] += count
    # acc
    acc_1 = [(correct_n / nun_class) for correct_n in total_correct_1]
    acc_2 = [(correct_n / nun_class) for correct_n in total_correct_2]
    acc_3 = [(correct_n / nun_class) for correct_n in total_correct_3]
    acc_4 = [(correct_n / nun_class) for correct_n in total_correct_4]
    acc_5 = [(correct_n / nun_class) for correct_n in total_correct_5]

    acc_n = [(correct_n / total_n) for correct_n in total_correct_n]
    acc_a = [(correct_a / total_a) for correct_a in total_correct_a]
    acc = [((total_correct_n[i] + total_correct_a[i]) / (total_n + total_a)) for i in range(len(threshold))]
    best_acc = np.max(acc)
    idx = np.argmax(acc)
    print("th=best.kcenter_acc；%f,uncertainty_acc:%f,adfl_acc:%f,adflk_acc:%f ,acc_knockoff:%f" % (
        acc_1[idx], acc_2[idx], acc_3[idx],
        acc_4[idx], acc_5[idx]))
    printout(0.95, acc_1, acc_2, acc_3, acc_4, acc_5, acc_n)
    printout(0.9, acc_1, acc_2, acc_3, acc_4, acc_5, acc_n)
    printout(0.8, acc_1, acc_2, acc_3, acc_4, acc_5, acc_n)
    printout(0.7, acc_1, acc_2, acc_3, acc_4, acc_5, acc_n)
    printout(0.6, acc_1, acc_2, acc_3, acc_4, acc_5, acc_n)
    printout(0.5, acc_1, acc_2, acc_3, acc_4, acc_5, acc_n)
    printout(0.4, acc_1, acc_2, acc_3, acc_4, acc_5, acc_n)
    printout(0.3, acc_1, acc_2, acc_3, acc_4, acc_5, acc_n)
    printout(0.2, acc_1, acc_2, acc_3, acc_4, acc_5, acc_n)
    printout(idx*0.01, acc_1, acc_2, acc_3, acc_4, acc_5, acc_n)
    # print(
    #     "th=0.95.uncertainty_acc:%f,kcenter_acc；%f,adfl_acc:%f,adflk_acc:%f ,de_acc_1:%f,de_acc_2:%f,de_acc_3:%f,de_acc_4:%f,bn_acc:%f" % (
    #         acc_1[95], acc_2[95], acc_3[95], acc_4[95], acc[95],))
    # print("th=0.9.uncertainty_acc:%f,kcenter_acc；%f,adfl_acc:%f,adflk_acc:%f all_acc:%f" % (
    #     acc_1[90], acc_2[90], acc_3[90], acc_4[90], acc[90]))
    # print("th=0.8.uncertainty_acc:%f,kcenter_acc；%f,adfl_acc:%f,adflk_acc:%f all_acc:%f" % (
    #     acc_1[80], acc_2[80], acc_3[80], acc_4[80], acc[80]))
    # print("th=0.7.uncertainty_acc:%f,kcenter_acc；%f,adfl_acc:%f,adflk_acc:%f all_acc:%f" % (
    #     acc_1[70], acc_2[70], acc_3[70], acc_4[70], acc[70]))
    # print("th=0.6.uncertainty_acc:%f,kcenter_acc；%f,adfl_acc:%f,adflk_acc:%f all_acc:%f" % (
    #     acc_1[60], acc_2[60], acc_3[60], acc_4[60], acc[60]))
    # print("th=0.5.uncertainty_acc:%f,kcenter_acc；%f,adfl_acc:%f,adflk_acc:%f all_acc:%f" % (
    #     acc_1[50], acc_2[50], acc_3[50], acc_4[50], acc[50]))
    # print("th=0.4.uncertainty_acc:%f,kcenter_acc；%f,adfl_acc:%f,adflk_acc:%f all_acc:%f" % (
    #     acc_1[40], acc_2[40], acc_3[40], acc_4[40], acc[40]))
    # print("th=0.3.uncertainty_acc:%f,kcenter_acc；%f,adfl_acc:%f,adflk_acc:%f all_acc:%f" % (
    #     acc_1[30], acc_2[30], acc_3[30], acc_4[30], acc[30]))
    # print("th=0.2.uncertainty_acc:%f,kcenter_acc；%f,adfl_acc:%f,adflk_acc:%f all_acc:%f" % (
    #     acc_1[20], acc_2[20], acc_3[20], acc_4[20], acc[20]))
    print("similarity:random:%f,uncertainty:%f,kcenter:%f,adfl:%f,adfl-k:%f" % (
        sim_r, sim_1 / 4, sim_2 / 4, sim_3 / 4, sim_4 / 4))

    best_threshold = idx * 0.01

    return best_acc, best_threshold, acc_n[idx], acc_a[idx], acc, acc_n, acc_a


def cal_score(model_front_d, model_front_ir, model_top_d, model_top_ir, normal_vec_front_d, normal_vec_front_ir,
              normal_vec_top_d, normal_vec_top_ir, test_loader_front_d, test_loader_front_ir, test_loader_top_d,
              test_loader_top_ir, score_folder, use_cuda):
    """
    Generate and save scores of top_depth/top_ir/front_d/front_ir views
    """
    assert int(len(test_loader_front_d)) == int(len(test_loader_front_ir)) == int(len(test_loader_top_d)) == int(
        len(test_loader_top_ir))
    total_batch = int(len(test_loader_front_d))
    sim_list = torch.zeros(0)
    sim_1_list = torch.zeros(0)
    sim_2_list = torch.zeros(0)
    sim_3_list = torch.zeros(0)
    sim_4_list = torch.zeros(0)
    label_list = torch.zeros(0).type(torch.LongTensor)
    for batch, (data1, data2, data3, data4) in enumerate(
            zip(test_loader_front_d, test_loader_front_ir, test_loader_top_d, test_loader_top_ir)):
        if use_cuda:
            data1[0] = data1[0].cuda()
            data1[1] = data1[1].cuda()
            data2[0] = data2[0].cuda()
            data2[1] = data2[1].cuda()
            data3[0] = data3[0].cuda()
            data3[1] = data3[1].cuda()
            data4[0] = data4[0].cuda()
            data4[1] = data4[1].cuda()

        assert torch.sum(data1[1] == data2[1]) == torch.sum(data2[1] == data3[1]) == torch.sum(data3[1] == data4[1]) == \
               data1[1].size(0)

        out_1 = model_front_d(data1[0])[1].detach()
        out_2 = model_front_ir(data2[0])[1].detach()
        out_3 = model_top_d(data3[0])[1].detach()
        out_4 = model_top_ir(data4[0])[1].detach()

        sim_1 = torch.mm(out_1, normal_vec_front_d.t())
        sim_2 = torch.mm(out_2, normal_vec_front_ir.t())
        sim_3 = torch.mm(out_3, normal_vec_top_d.t())
        sim_4 = torch.mm(out_4, normal_vec_top_ir.t())
        sim = (sim_1 + sim_2 + sim_3 + sim_4) / 4

        sim_list = torch.cat((sim_list, sim.squeeze().cpu()))
        label_list = torch.cat((label_list, data1[1].squeeze().cpu()))
        sim_1_list = torch.cat((sim_1_list, sim_1.squeeze().cpu()))
        sim_2_list = torch.cat((sim_2_list, sim_2.squeeze().cpu()))
        sim_3_list = torch.cat((sim_3_list, sim_3.squeeze().cpu()))
        sim_4_list = torch.cat((sim_4_list, sim_4.squeeze().cpu()))
        print(f'Evaluating: Batch {batch + 1} / {total_batch}')

    np.save(os.path.join(score_folder, 'score_front_d.npy'), sim_1_list.numpy())
    print('score_front_d.npy is saved')
    np.save(os.path.join(score_folder, 'score_front_IR.npy'), sim_2_list.numpy())
    print('score_front_IR.npy is saved')
    np.save(os.path.join(score_folder, 'score_top_d.npy'), sim_3_list.numpy())
    print('score_top_d.npy is saved')
    np.save(os.path.join(score_folder, 'score_top_IR.npy'), sim_4_list.numpy())
    print('score_top_IR.npy is saved')
