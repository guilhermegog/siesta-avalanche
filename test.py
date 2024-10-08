from benchmarks.siesta_benchmark import make_cl_benchmark
from avalanche.benchmarks.classic import SplitImageNet
import numpy as np


def main():
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
        end_ix = start+end

        exp_dict[i] = result[start:end_ix]
        start = end_ix
    n_exp = 19
    start = 0
    for i in range(n_exp):
        if i == 0:
            end = 100
        else:
            end = 50

        exp_dict[i] = result[start:start+end]
        exp_dict = {}
        exp_dict[0] = 100
        start = end_ix
    benchmark = SplitImageNet(dataset_root="/space/gguedes/datasets/imagenet1k",
                              n_experiences=19,
                              per_exp_classes=exp_dict,
                              fixed_class_order=result,
                              return_task_id=True,
                              seed=42,
                              )

    print(benchmark.n_classes_per_exp)
    print(benchmark.n_experiences)


if __name__ == "__main__":
    main()
