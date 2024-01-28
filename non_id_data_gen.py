import numpy as np

def random_indices(arr, k):
    indices = np.arange(len(arr))
    random_indices = np.random.choice(indices, size=k, replace=False)
    return random_indices

def select_labels_indices(labels, num_samples):
    num_labels = len(labels)
    random_indices = np.random.choice(range(num_labels), size=num_samples, replace=False)

    return random_indices

def dirichlet_split_noniid(train_labels):

    n_classes = train_labels.max()+1
    aa=np.repeat(0.5, 1000)
    label_distribution = np.random.dirichlet(aa)
    label_distribution = label_distribution / label_distribution.sum()
    label_distribution = ((label_distribution)*2000).astype(int)
    append_list=[0]*1000
    for i in range(2000-label_distribution.sum()):
        t=np.random.randint(1000)
        append_list[t]=append_list[t]+1
    label_distribution=label_distribution+append_list

    # print(label_distribution.sum())
    class_idcs = [np.argwhere(train_labels==y).flatten()
           for y in range(n_classes)]


    client_idcs = []

    for c, fracs in zip(class_idcs, label_distribution):
        if(fracs>0):
            indices = random_indices(c, fracs)
            client_idcs.append(c[indices])
    ar=np.concatenate(client_idcs)
    np.random.shuffle(ar)
    return ar

if __name__ == "__main__":

    your_data="" #load your data
    your_label ="" #load your label
    for i in range(400):
        t = dirichlet_split_noniid(your_label)
        np.save("./non_id_data/" + str(i) + ".npy", your_data)








