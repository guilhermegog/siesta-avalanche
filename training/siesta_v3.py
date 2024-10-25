import random
import numpy as np
from collections import defaultdict
from typing import Callable, Optional, Union

import torch
import copy
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
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
    :param sleep_frequency: The frequency at which the sleep phase is initiated #experiences
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
        num_classes: int = 1000,
        criterion=torch.nn.CrossEntropyLoss(),
        lr: float = 0.001,
        tau: float = 1.0,
        seed: Optional[int] = None,
        sleep_frequency: int = 2,
        sleep_mb_size: int = 512,
        eval_mb_size: int = 128,
        memory_size: int = 959665,
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

        self.classifier_G, self.classifier_F, self.aol = create_classifiers()
        self.classifier_G.cuda()
        self.classifier_F.cuda()
        self.sleep_criterion = criterion.cuda()
        self.lr = lr
        self.tau = tau
        self.sleep_mb_size = sleep_mb_size
        self.memory_size = memory_size
        self.sleep_n_iter = int(20017 * (64 / self.sleep_mb_size))
        self.num_classes = num_classes
        self.sleep_frequency = sleep_frequency
        self.rotation = 0
        # Register buffer with mem_size x latent_dim samples
        buffer = get_buffer()

        self.latent_dict = buffer["latent_dict"]
        self.rehearsal_ixs = buffer["rehearsal_ixs"]
        self.class_id_to_item_ix_dict = buffer["class_id_to_item_ix_dict"]
        self.recent_class_list = buffer["recent_class_list"]
        self.total_class_list = buffer["recent_class_list"]
        self.curr_buff = len(self.rehearsal_ixs)
        model = torch.nn.Sequential(self.classifier_G,
                                    self.classifier_F)

        super(SIESTA, self).__init__(
            model=model,
            optimizer=None,  # No optimizer for SIESTA online updates
            criterion=criterion,
            train_mb_size=512,
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
    def store_sample(self, feat, labels):
        # Go through the batched features
        # Changed the unique item ix to
        # a sequential class counter to facilitate
        # siesta implementation
        for x, y in zip(feat, labels):
            self.latent_dict[self.curr_buff] = [x.cpu().numpy(), y.cpu().numpy()]
            self.rehearsal_ixs.append(self.curr_buff)
            self.class_id_to_item_ix_dict[int(
                y.cpu().numpy())].append(self.curr_buff)
            if self.curr_buff >= self.memory_size:
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
                self.curr_buff += 1

    @torch.no_grad()
    def sample_memory(self):
        ixs = torch.empty(len(self.rehearsal_ixs), dtype=torch.long)
        labels = torch.empty(len(self.rehearsal_ixs), dtype=torch.long)
        for ii, v in enumerate(self.rehearsal_ixs):
            labels[ii] = torch.from_numpy(self.latent_dict[v][1])
            ixs[ii] = v

        labels = labels.numpy()
        ixs = ixs.numpy()
        class_list = np.unique(labels)
        print(class_list)
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
        np.random.shuffle(replay_idxs)
        return replay_idxs

    def sleep_training(self, sleep_mb_size):
        # Sample uniformly from the replay memory_size

        total_loss = CMA()
        params = self.get_layerwise_params(self.classifier_F, self.lr)
        optimizer = SGD(params, lr=self.lr)

        classifier_F = self.classifier_F.cuda()
        classifier_F.train()
        total_stored_samples = len(self.rehearsal_ixs)
        print("Number of stored samples: ", total_stored_samples)
        batch = self.sleep_mb_size
        num_iter = self.sleep_n_iter
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, steps_per_epoch=int(num_iter), epochs=1)

        replay_ids = self.sample_memory()
        total_stored_samples = len(replay_ids)

        features = np.empty((len(replay_ids), 80, 14, 14), dtype=np.float32)
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
            feat_batch = torch.from_numpy(features[start:end])
            assert feat_batch.nelement() != 0, f"Batch is empty at start: {start}, end: {end}"
            labels_batch = labels[start:end].cuda()
            output = classifier_F(feat_batch.cuda())

            loss = self.sleep_criterion(output, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss.update(loss.item())

            if (i+1) % 5000 == 0 or i == 0 or (i+1) == num_iter:
                print("Iter:", (i+1), "-- Loss: %1.5f" % total_loss.avg)

    def online_update(self, z, label, recent_labels):
        recent_class_list = np.unique(recent_labels.numpy())
        label = label.squeeze()
        for x, y in zip(z, label):
            self.aol.fit(x, y.view(1,))

    def forward(self, sleep=False):
        """Compute the model's output given the current mini-batch."""

        with torch.no_grad():
            self.model.eval()
            self.classifier_G.eval()
            self.classifier_F.eval()
            feat = self.classifier_G(self.mb_x)
            penult_feat = self.classifier_F.get_penultimate_feature(feat)
            output = self.classifier_F.model.classifier[1](penult_feat)
            output = self.classifier_F.model.classifier[2](output)
            output = self.classifier_F.model.classifier[3](output)
        return output, penult_feat, feat

    def eval_forward(self):
        """Compute the model's output given the current mini-batch."""

        self.model.eval()
        self.classifier_G.eval()
        self.classifier_F.eval()
        feat = self.classifier_G(self.mb_x)
        out = self.classifier_F(feat)
        return out

    def _before_training_exp(self, **kwargs):
        """Setup to train on a single experience."""
        # Data Adaptation (e.g. add new samples/data augmentation)
        self._before_train_dataset_adaptation(**kwargs)
        self.train_dataset_adaptation(**kwargs)
        self._after_train_dataset_adaptation(**kwargs)

        self.make_train_dataloader(**kwargs)

        # Model Adaptation (e.g. freeze/add new units)
        self.check_model_and_optimizer(**kwargs)
        super()._before_training_exp(**kwargs)

    def training_epoch(self, **kwargs):
        n_exp = self.clock.train_exp_counter
        recent_labels = []
        start = 0
        if(n_exp != 0):
            self.rotation += 1
            for mb_it, self.mbatch in enumerate(self.dataloader):
                self._unpack_minibatch()
                self._before_training_iteration(**kwargs)
                self.loss = self._make_empty_loss()
                end = start + self.mb_y.shape[0]
                recent_labels.append(self.mb_y.squeeze().cpu().numpy())
                start = end
                # Forward pass
                self._before_forward(**kwargs)
                self.mb_output, penult_feature, feature = self.forward()
                self.store_sample(feature, self.mb_y)
                self._after_forward(**kwargs)

                # Loss computation
                self.loss = self.criterion()

                # Optimization step
                self._before_update(**kwargs)
                self.online_update(penult_feature, self.mb_y,
                                   self.mb_y.squeeze().cpu())

                self._after_update(**kwargs)

                self._after_training_iteration(**kwargs)

            if(self.rotation - self.sleep_frequency == 0):
                self.sleep_training(self.sleep_mb_size)
                self.rotation = 0

            recent_class_list = np.unique(np.concatenate(recent_labels))
            bias = torch.ones(1).cuda()
            for k in recent_class_list:
                k = torch.tensor(k, dtype=torch.int32)
                mu_k = self.aol.grab_mean(k)
                self.classifier_F.state_dict(
                )["model.classifier.3.weight"][k] = mu_k
                self.classifier_F.state_dict(
                )["model.classifier.3.bias"][k] = bias
        else:
            pass

    def eval_epoch(self, **kwargs):
        """Evaluation loop over the current `self.dataloader`."""
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_eval_iteration(**kwargs)

            self._before_eval_forward(**kwargs)
            self.mb_output = self.eval_forward()
            self._after_eval_forward(**kwargs)
            self.loss = self.criterion()

            self._after_eval_iteration(**kwargs)

    def make_optimizer(self, **kwargs):
        """SIESTA online updates require no optimizer."""
        pass


__all__ = ["SIESTA"]
