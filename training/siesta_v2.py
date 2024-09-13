import random
import warnings
from collections import defaultdict
from typing import Callable, Optional, Union

import torch
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.templates.strategy_mixin_protocol import CriterionType
from torch.optim import SGD
from torch.utils.data import DataLoader

from utils import build_model
from model.online_learner import RunningMean


class SIESTA(SupervisedTemplate):
    """
    SIESTA pre_alpha implementation, currently implementing the model embedded in the
    strategy itslef, this will be changed in future iterations to allow for the
    flexibility of the 3 distinct segments of the network (H,G,F).
    :param num_classes: The number of classes in the dataset, assumes siesta_net model
    :param criterion: The loss function used to train the model, defaults to nll_loss
    :param optimizer: The optimizer used to train the model
    :param lr: The learning rate used to train the model
    :param tau: The temperature parameter used in the softmax output layer check SiestaNet for more details
    :param seed: The seed used to initialize the model
    :param sleep_frequency: The frequency at which the sleep phase is initiated #classes
    :param sleep_mb_size: The size of the minibatch used during the sleep phase
    :param sleep_n_iter: The number of iterations used during the sleep phase
    # samples (to be changed to Bytes)
    :param memory_size: The size of the memory buffer
    :param device: The device used to train the model, defaults to cpu
    :param evaluator: The evaluator used to evaluate the model
    :param eval_every: The frequency at which the model is evaluated

    """

    def __init__(
        self,
        *,
        num_classes: int = 100,
        criterion: CriterionType = torch.nn.CrossEntropyLoss(),
        lr: float = 0.001,
        tau: float = 1.0,
        seed: Optional[int] = None,
        sleep_frequency: int = 10,
        sleep_mb_size: int = 32,
        sleep_n_iter: int = 1000,
        eval_mb_size: int = 32,
        memory_size: int = 100000000,
        device: Union[str, torch.device] = "cpu",
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every: int = -1,
        **kwargs,
    ):

        # As of this implementation the SIESTA strategy only supports the use
        # of the specific model implemented in SiestaNet. Aditional configurability
        # in the model implementation is require to define where to freeze and extract
        # LRs from.

        model = build_model(8)
        self.sleep_criterion = criterion
        self.lr = lr
        self.tau = tau
        self.sleep_mb_size = sleep_mb_size
        self.sleep_n_iter = sleep_n_iter
        self.memory_size = memory_size
        self.num_classes = num_classes
        # Register buffer with mem_size x latent_dim samples
        self.replay_memory = defaultdict(list)
        self.sleep_frequency = sleep_frequency
        self.online_learner = RunningMean(2000, num_classes)

        super(SIESTA, self).__init__(
            model=model,
            optimizer=None,  # No optimizer for SIESTA online updates
            criterion=criterion,
            train_mb_size=1,
            train_epochs=1,
            eval_mb_size=eval_mb_size,
            plugins=[],
            device=device,
            evaluator=evaluator,
            eval_every=eval_every,
            **kwargs,
        )

    @torch.no_grad()
    def store_sample(self, lr, label):
        # Also need to include OPQ to quantize vectors in
        current_class = self.replay_memory[label]
        total_samples = torch.sum(current_class)
        if total_samples == self.memory_size:
            # find class with most samples, remove one randomly
            max_samples = max(self.replay_memory.values(), key=len)
            max_label = next(
                key for key, value in self.replay_memory.items() if value == max_samples)
            random_sample = random.randrange(
                0, len(self.replay_memory[max_label]))
            self.replay_memory[max_label].pop(random_sample)

        label = label.item()
        self.replay_memory[label].append(lr)

    @torch.no_grad()
    def sample_memory(self, mb_size):
        minibatch = []
        labels = list(self.replay_memory.keys())
        n_labels = len(labels)

        samples_per_label = mb_size // n_labels
        if samples_per_label == 0:
            print("What happened")
            print("mb_size", mb_size)
            print("n_labels", n_labels)
            print("Here are the labels!", labels)

        # Uniform Sampling
        for label in labels:
            minibatch.extend(
                random.sample(
                    self.replay_memory[label],
                    min(samples_per_label, len(self.replay_memory[label])),
                )
            )
        samples = torch.stack(minibatch)
        samples = torch.squeeze(samples, dim=1)
        labels = [label for label in labels]
        minibatch_labels = torch.tensor(labels).repeat(samples_per_label)[
            : samples.shape[0]
        ]

        return samples, minibatch_labels

    def sleep_training(self, sleep_mb_size):
        # Sample uniformly from the replay memory_size
        optimizer = SGD(self.model.parameters(), lr=self.lr)

        print("-------->> Sleep Phase Initiated <<--------")
        for i in range(self.sleep_n_iter):
            optimizer.zero_grad()
            mb_x, mb_y = self.sample_memory(sleep_mb_size)
            mb_x, mb_y = mb_x.to(self.device), mb_y.to(self.device)
            mb_out = self.model.get_g_net(mb_x)
            mb_out = self.model.get_f_net(mb_out)
            sleep_loss = self.sleep_criterion(mb_out, mb_y)
            if (i % 100) == 0:
                print(f"Loss at iteration {i}: {sleep_loss.item()}")
            sleep_loss.backward()
            optimizer.step()

        print(
            f"-------->> Sleep Phase Complete, {self.sleep_n_iter} <<--------")
        print("Loss after sleep phase:", sleep_loss)

    def online_update(self, z, label):
        self.online_learner.fit(z, label)
        mean = self.online_learner.grab_mean(label)
        with torch.no_grad():
            self.model.f_net.state_dict(
            )["3.weight"][label].copy_(mean)

    def forward(self, sleep=False):
        """Compute the model's output given the current mini-batch."""

        self.model.eval()
        lr = self.model.get_h_net(self.mb_x)
        z = self.model.get_g_net(lr)
        out = self.model.get_f_net(z)
        return out, z, lr

    def _before_training_exp(self, **kwargs):

        labels = list(self.replay_memory.keys())
        n_labels = len(labels)
        if n_labels % self.sleep_frequency == 0 and n_labels != 0:
            self.sleep_training(self.sleep_mb_size)

        super()._before_training_exp(**kwargs)

    def make_train_dataloader(
        self, num_workers=0, shuffle=True, persistent_workers=False, **kwargs
    ):
        """
        How to implement this as two dataloaders?
        I would require a sleep dataloader that is ued to train the model during sleep phase, aka with a given batch size
        But I would also require a dataloader that operates during the awake phase, essentially providing a single minibatch per iteration
        ACTUALLY, I think a single dataloader is required, as the second dataloader will simply be a byproduct of the memory buffer,
        """
        # The data loader will only work in the awake phase. As such it will only load a single sample from an experience
        # This is because the sleep phase will be handled by the memory buffer, in a seperate step
        assert self.adapted_dataset is not None, "No dataset has been provided."

        collate_fn = (
            self.adapted_dataset.collate_fn
            if hasattr(self.adapted_dataset, "collate_fn")
            else None
        )

        other_dataloader_args = self._obtain_common_dataloader_parameters(
            batch_size=1,
            persistent_workers=persistent_workers,
            num_workers=num_workers,
            shuffle=shuffle,
            **kwargs,
        )

        self.dataloader = DataLoader(
            self.adapted_dataset,
            collate_fn=collate_fn,
            **other_dataloader_args,
            # Inspired by the AR1 implementation (maybe this is actually not fully required...)
        )

    def training_epoch(self, **kwargs):
        for mb_it, self.mbatch in enumerate(self.dataloader):
            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)
            self.loss = self._make_empty_loss()

            # Forward pass
            self._before_forward(**kwargs)
            self.mb_output, z, latent_act = self.forward()

            self.store_sample(latent_act, self.mb_y)
            self._after_forward(**kwargs)

            # Loss computation
            self.loss = self.criterion()

            # Optimization step
            self._before_update(**kwargs)
            self.online_update(z, self.mb_y)
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def eval_epoch(self, **kwargs):
        """Evaluation loop over the current `self.dataloader`."""
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_eval_iteration(**kwargs)

            self._before_eval_forward(**kwargs)
            self.mb_output, z, latent_act = self.forward()
            self._after_eval_forward(**kwargs)
            self.loss = self.criterion()

            self._after_eval_iteration(**kwargs)

    def make_optimizer(self, **kwargs):
        """SIESTA online updates require no optimizer."""
        pass


__all__ = ["SIESTA"]
