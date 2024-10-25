from benchmarks.imagenet_utils import get_imagenet_dataset
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import DataAttribute, make_avalanche_dataset
import time
import numpy as np


def make_avl_datasets(images_dir, label_dir):
    first_exp = 100
    exp = 50
    i = 0
    t1 = time.time()
    while i < 1000:
        if(i == 0):
            train_ds = get_imagenet_dataset(
                dirname=images_dir + '/' + 'train', label_dir=label_dir, split="train", min_class=i, max_class=first_exp, return_item_ix=True
            )
            test_ds = get_imagenet_dataset(
                dirname=images_dir + '/' + 'val', label_dir=label_dir, split="val", min_class=i, max_class=first_exp
            )
            x, y, ixs = train_ds[:]
            da_train = DataAttribute(train_ds.y, "targets")
            avl_train_ds = make_avalanche_dataset(
                train_ds, data_attributes=da_train)
            avl_test_ds = make_avalanche_dataset(test_ds)
            i = i + first_exp
        else:
            train_ds = get_imagenet_dataset(
                dirname=images_dir + '/' + 'train', label_dir=label_dir, split="train", min_class=i, max_class=i+exp, return_item_ix=True
            )
            test_ds = get_imagenet_dataset(
                dirname=images_dir + '/' + 'val', label_dir=label_dir, split="val", min_class=i, max_class=i+exp
            )
            x, y, ixs = train_ds[:]
            da_train = DataAttribute(train_ds.data, "targets")
            train_ds = make_avalanche_dataset(
                train_ds, data_attributes=da_train)
            test_ds = make_avalanche_dataset(test_ds)
            avl_train_ds = avl_train_ds.concat(train_ds)
            avl_test_ds = avl_test_ds.concat(test_ds)
            i = i + exp
    t2 = time.time()
    print("Succesfully instantiated imagenet, time: ", t2-t1)
    return avl_train_ds, avl_test_ds


def make_cl_benchmark():
    label_dir = '/space/gguedes/Documents/siesta-avalanche/benchmarks/'
    arr = np.load(label_dir + '/imagenet_files/imagenet_train_labels.npy')
    result = arr[np.insert(np.diff(arr) != 0, 0, True)]

    exp_dict = {}
    n_exp = 19
    start = 0
    for i in range(n_exp):
        if i == 0:
            end = 100
        else:
            end = 50

        exp_dict[i] = result[start:start+end]
        start = end

    train_ds, test_ds = make_avl_datasets(
        images_dir='/space/gguedes/datasets/imagenet1k', label_dir=label_dir)

    my_scenario = nc_benchmark(
        train_dataset=train_ds,
        test_dataset=test_ds,
        task_labels=True,
        n_experiences=19,
        shuffle=False,
        per_exp_classes=exp_dict,
    )
    return my_scenario
