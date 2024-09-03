from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.templates.strategy_mixin_protocol import CriterionType
from torch.utils.data import DataLoader

from model.siesta_net import SiestaNet


class SIESTA(SupervisedTemplate):
    def __init__(
        self,
        loss: CriterionType = torch.nn.functional.nll_loss(),
        optimizer: torch.optim.Optimizer = None,  # Optimizer for sleep phase
        lr: float = 0.001,
        tau=1.0,
        seed: Optional[int] = None,
        sleep_frequency: int = 10,
        sleep_mb_size: int = 32,
        sleep_n_iter: int = 100,
        memory_size: int = 1000,
        num_classes: int = 100,
        device=None,
    ):

        # As of this implementation the SIESTA strategy only supports the use
        # of the specific model implemented in SiestaNet. Aditional configurability
        # in the model implementation is require to define where to freeze and extract
        # LRs from.

        model = SiestaNet(num_classes=num_classes)
        self.loss = loss
        self.sleep_loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.tau = tau
        self.sleep_mb_size = sleep_mb_size
        self.sleep_n_iter = sleep_n_iter
        self.memory_size = memory_size
        self.num_classes = num_classes
        # Register buffer with mem_size x latent_dim samples
        self.register_buffer("stored_samples_nr", torch.zeros(memory_size))
        self.replay_memory = defaultdict(list)
        self.sleep_frequency = sleep_frequency

        super().__init__(
            model=model,
            optimizer=optimizer,
            loss=loss,
            device=device,
        )

        def store_sample(self, lr, label):
            # Also need to include OPQ to quantize vectors in
            if self.stored_samples_nr == self.memory_size:
                # FIFO behaviour, this needs to be changed
                self.replay_memory[label].pop(0)

            self.replay_memory[label].append(lr)
            total_items = sum([len(v) for v in self.replay_memory.values()])
            self.stored_samples_nr = torch.tensor(total_items)

        def sample_memory(self, mb_size):
            minibatch = []
            labels = list(self.replay_memory.keys())
            n_labels = len(labels)

            samples_per_label = sleep_mb_size // n_labels

            # Uniform Sampling
            for label in labels:
                minibatch.extend(
                    torch.random.sample(
                        self.replay_memory[label],
                        min(samples_per_label, len(self.memory_buffer[label])),
                    )
                )
            samples = torch.stack(minibatch)
            minibatch_labels = torch.tensor(labels).repeat(samples_per_label)[
                : samples.shape[0]
            ]

            return samples, minibatch_labels

        def sleep_training(self, sleep_mb_size):
            # Sample uniformly from the replay memory_size
            for i in range(sleep_n_iter):
                print("-------->> Initiating Sleep Phase <<--------")
                self.optimizer.zero_grad()
                mb_x, mb_y = sample_memory(sleep_mb_size)
                mb_out = self.model(mb_x, sleep=True)
                self.sleep_loss = self.sleep_loss(torch.log(mb_out), self.mb_y)
                self.sleep_loss.backward()
                self.optimizer.step()

            print("-------->> Sleep Phase Complete <<--------")
            print("Loss after sleep phase:", self.sleep_loss.item())

        def _before_training_exp(self, **kwargs):

            if self.clock.train_exp_counter % self.sleep_frequency == 0:
                sleep_training(self.sleep_mb_size)

            super._before_training_exp(**kwargs)

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
                **kwargs
            )

            self.dataloader = DataLoader(
                self.adapted_dataset,
                collate_fn=collate_fn,
                **other_dataloader_args
                # Inspired by the AR1 implementation (maybe this is actually not fully needed...)
            )

        def training_epoch(self, **kwargs):
            for mb_it, self.mbatch in enumerate(self.dataloader):
                self._unpack_minibatch(self.mbatch)
                self._before_training_iteration(**kwargs)

                self.optimizer.zero_grad()

                self._before_forward(**kwargs)
                self.mb_output, self.z, self.lr = self.model(self.mb_x, sleep=False)
                store_sample(self.lr, self.mb_y)
                self.loss = self.loss(torch.log(self.mb_output), self.mb_y)

                self._after_forward(**kwargs)
                self._after_loss_backward(**kwargs)
                # Optimization step
                self._before_update(**kwargs)
                self.model.f_classifier.online_update(self.z, self.mb_y)
                self._after_update(**kwargs)

                self._after_training_iteration(**kwargs)
