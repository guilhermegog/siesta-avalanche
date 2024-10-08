import random
import numpy as np
from torch.nn import optim
from collections import defaultdict
from typing import Callable, Optional, Union

import torch
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.templates.strategy_mixin_protocol import CriterionType
from torch.optim import SGD
from torch.utils.data import DataLoader
from imagenet_utils import get_imagenet_data_loader


from training.create_mobnet import create_classifiers, get_buffer, CMA


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
        num_classes: int = 1000,
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

        self.classifier_F, self.classifier_G, self.aol = create_classifiers()
        self.sleep_criterion = criterion
        self.lr = lr
        self.tau = tau
        self.sleep_mb_size = sleep_mb_size
        self.sleep_n_iter = sleep_n_iter
        self.memory_size = memory_size
        self.num_classes = num_classes
        # Register buffer with mem_size x latent_dim samples
        buffer = get_buffer()

        self.latent_dict = buffer["latent_dict"]
        self.rehearsal_ixs = buffer["rehearsal_ixs"]
        self.class_id_to_item_ix_dict = buffer["class_id_to_item_ix_dict"]
        self.recent_class_list = buffer["recent_class_list"]
        self.total_class_list = buffer["recent_class_list"]

        super(SIESTA, self).__init__(
            self,
            num_classes=1000,
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

    def get_layerwise_params(self, classifier, lr):
        trainable_params = []
        layer_names = []
        lr_mult = 0.99  # 0.99
        for idx, (name, param) in enumerate(classifier.named_parameters()):
            layer_names.append(name)
        # reverse layers
        layer_names.reverse()
        # store params & learning rates
        for idx, name in enumerate(layer_names):
            # append layer parameters
            trainable_params += [{'params': [p for n, p in classifier.named_parameters() if n == name and p.requires_grad],
                                  'lr': lr}]
            # update learning rate
            lr *= lr_mult
        return trainable_params

    @torch.no_grad()
    def store_sample(self, feat, labels, item_ixs):

        for x, y, item_ix in zip(feat, labels, item_ixs):
            self.latent_dict[int(item_ix.numpy())] = [x, y.numpy()]
            self.rehearsal_ixs.append(int(item_ix.numpy()))
            self.class_id_to_item_ix_dict[int(y.numpy())].append(
                int(item_ix.numpy()))
            if self.count >= self.max_buffer_size:
                max_key = max(
                    self.class_id_to_item_ix_dict,
                    key=lambda x: len(self.class_id_to_item_ix_dict[x]),
                )
                max_class_list = self.class_id_to_item_ix_dict[max_key]
                rand_item_ix = random.choice(max_class_list)
                max_class_list.remove(rand_item_ix)
                self.latent_dict.pop(rand_item_ix)
                self.rehearsal_ixs.remove(rand_item_ix)
            else:
                self.count += 1

    @torch.no_grad()
    def sample_memory(self):
        ixs = torch.empty(len(self.rehearsal_ixs), dtype=torch.long)
        labels = torch.empty(len(self.rehearsal_ixs), dtype=torch.long)
        for ii, v in enumerate(self.rehearsal_ixs):
            labels[ii] = torch.from_numpy(self.latent_dict[v][1])
            ixs[ii] = v

        class_list = np.unique(labels)
        replay_idxs = []
        k = 1
        count = 0
        budget = self.sleep_mb_size * self.sleep_n_iter
        while(count < budget):
            for c in class_list:
                ixs_current_class = ixs[labels == c]
                sel_idx = np.random.choice(
                    ixs_current_class, size=k, replace=False)
                count += k
                replay_idxs.append(torch.from_numpy(sel_idx))
                if(count >= budget):
                    break

        replay_idxs = torch.cat(replay_idxs, dim=0)
        replay_idxs = torch.tensor(replay_idxs[:budget], dtype=torch.int32)
        print("Number of samples selected for rehearsal ", len(replay_idxs))
        assert len(replay_idxs) <= budget
        replay_idxs = np.array(replay_idxs, dtype=np.int32)
        np.random_shuffle(replay_idxs)
        return replay_idxs

    def sleep_training(self, sleep_mb_size):
        # Sample uniformly from the replay memory_size
        sleep_acc_all5 = []
        total_loss = CMA()
        params = self.get_layerwise_params(self.classifier_F, self.sleep_lr)
        optimizer = SGD(params, lr=self.lr)

        classifier_F = self.classifier_F.cuda()
        classifier_F.train()
        total_stored_samples = len(self.rehearsal_ixs)
        print("Number of stored samples: ", total_stored_samples)
        batch = self.sleep_mb_size
        num_iter = self.sleep_n_iter
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, steps_per_epoch=num_iter, epochs=1)

        replay_ids = self.sample_memory()

        features = np.empty((len(replay_ids), 80, 14, 14), dtype=np.uint8)
        labels = torch.empty((len(replay_ids)), dtype=torch.long).cuda()
        for ii, v in enumerate(replay_ids):
            v = v.item()
            features[ii] = self.latent_dict[v][0]
            labels[ii] = torch.from_numpy(self.latent_dict[v][1])

        for i in range(num_iter):
            start = i*batch
            if(start > (total_stored_samples-batch)):
                end = total_stored_samples
            else:
                end = (i+1)*batch
            feat_batch = features[start:end]
            labels_batch = labels[start:end].cuda()
            output = classifier_F(feat_batch.cuda())

            loss = self.criterion(output, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss.update(loss.item())

            if (i+1) % 5000 == 0 or i == 0 or (i+1) == num_iter:
                print("Iter:", (i+1), "-- Loss: %1.5f" % total_loss.avg)

    def online_update(self, z, label):
        self.online_learner.fit(z, label)
        mean = self.online_learner.grab_mean(label)
        with torch.no_grad():
            self.model.f_net.state_dict(
            )["3.weight"][label].copy_(mean)

    def forward(self, sleep=False):
        """Compute the model's output given the current mini-batch."""

        self.model.eval()
        feat = self.classifier_G
        out = self.classifier_F
        return out, feat

    def _before_training_exp(self, **kwargs):

        labels = list(self.replay_memory.keys())
        n_labels = len(labels)
        if n_labels % self.sleep_frequency == 0 and n_labels != 0:
            self.sleep_training(self.sleep_mb_size)

        super()._before_training_exp(**kwargs)

    def get_data_loader(
        self,
        images_dir,
        label_dir,
        split,
        min_class,
        max_class,
        batch_size=128,
        return_item_ix=False,
    ):

        data_loader = get_imagenet_data_loader(
            images_dir + "/" + split,
            label_dir,
            split,
            batch_size=batch_size,
            shuffle=False,
            min_class=min_class,
            max_class=max_class,
            return_item_ix=return_item_ix,
        )
        return data_loader

    def training_epoch(self, **kwargs):
        for mb_it, self.mbatch in enumerate(self.dataloader):
            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)
            self.loss = self._make_empty_loss()

            # Forward pass
            self._before_forward(**kwargs)
            self.mb_output, feature = self.forward()

            self.store_sample(feature, self.mb_y)
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
