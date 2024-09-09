import torch
import torch.nn as nn
from avalanche.benchmarks import SplitCIFAR100

from training.siesta import SIESTA


def main():
    device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")

    # Define transforms for the CIFAR-100 dataset
    criterion = nn.CrossEntropyLoss()
    benchmark = SplitCIFAR100(n_experiences=10, return_task_id=False)

    strategy = SIESTA(
        num_classes=100,
        criterion=criterion,
        device=device,
        memory_size=100000,
        sleep_mb_size=128,
        sleep_n_iter=int(10e3),
    )

    # Train the model
    results = []
    for experience in benchmark.train_stream:
        print(f"Start of experience: {experience.current_experience}")

        strategy.train(experiences=experience)

        print(f"End of experience: {experience.current_experience}")
        if experience.current_experience != 0:
            if experience.current_experience + 1 % 10 == 0:
                results.append(strategy.eval(benchmark.test_stream))


if __name__ == "__main__":
    main()
