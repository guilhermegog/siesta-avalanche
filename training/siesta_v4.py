import random
from collections import defaultdict
from typing import Callable, Optional, Union

import numpy as np
import torch
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
from torch.optim import SGD

from training.create_mobnet import get_buffer

# Implementation with model + benchmark flexibility


class OnlineLearner(torch.nn.Module):
    def __init__(self, num_classes, latent_dim, device):

        super(OnlineLearner, self).__init__()

        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.device = device
        self.mu_k = torch.zeros((num_classes, latent_dim), dtype=torch.float64).to(
            self.device
        )
        self.c_k = torch.zeros((num_classes), dtype=int).to(self.device)

    @torch.no_grad()
    def fit(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)

        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(y.shape) < 2:
            y = y.unsqueeze(0)

        # update class means
        self.mu_k[y, :] += (x - self.mu_k[y, :])/(self.c_k[y] + 1).unsqueeze(1)
        self.c_k[y] += 1

    def grab_mean(self, y):
        return self.mu_k[y]


class SIESTA(SupervisedTemplate):
    """
    SIESTA pre_alpha implementation, currently implementing the model embedded in the
    strategy itslef, this will be changed in future iterations to allow for the
    flexibility of the 3 distinct segments of the network (H,G,F).
    :param classifier_G: The feature extractor of the model, this is the point at which the
    features will get extracted from for replay
    :param classifier_F: The classifier of the model, this is the point at which the classifier
    will be updated with the new classes. Must include a weight layer at the last layer to
    capture prototypes.
    :param pretrained: A flag to indicate if the model is pretrained, defaults to False. If True
    requires the checkpoint. Optionally can also include a
    :param num_classes: The number of classes in the dataset, assumes siesta_net model
    :param criterion: The loss function used to train the model, defaults to CrossEntropyLoss
    :param optimizer: The optimizer used to train the model, defaults to SGD
    :param lr: The learning rate used to during sleep phase for the model
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
        classifier_G: torch.nn.Module,
        classifier_F: torch.nn.Module,
        pretrained: bool = False,
        num_classes: int = 1000,
        embed_size: int = 1024,
        criterion=torch.nn.CrossEntropyLoss(),
        lr: float = 0.001,
        tau: float = 1.0,
        seed: Optional[int] = None,
        sleep_frequency: int = 2,
        sleep_n_iter: int = 25000,
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

        self.classifier_G, self.classifier_F = classifier_G, classifier_F
        self.model = torch.nn.Sequential(self.classifier_G, self.classifier_F)
        print(embed_size)
        self.aol = OnlineLearner(num_classes, embed_size, device)
        self.classifier_G.cuda()
        self.classifier_F.cuda()
        self.sleep_criterion = criterion.cuda()
        self.lr = lr
        self.tau = tau
        self.sleep_mb_size = sleep_mb_size
        self.memory_size = memory_size
        self.sleep_n_iter = sleep_n_iter
        self.num_classes = num_classes
        self.sleep_frequency = sleep_frequency
        self.rotation = 0
        # Register buffer with mem_size x latent_dim samples
        if not pretrained:
            self.latent_dict = {}
            self.rehearsal_ixs = []
            self.class_id_to_item_ix_dict = defaultdict(list)
            self.recent_class_list = []
            self.total_class_list = []
            self.curr_buff = 0
        else:
            buffer = get_buffer()
            self.latent_dict = buffer["latent_dict"]
            self.rehearsal_ixs = buffer["rehearsal_ixs"]
            self.class_id_to_item_ix_dict = buffer["class_id_to_item_ix_dict"]
            self.recent_class_list = buffer["recent_class_list"]
            self.total_class_list = buffer["recent_class_list"]
            self.curr_buff = len(self.rehearsal_ixs)

        super(SIESTA, self).__init__(
            model=self.model,
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

    # Needed for sleep strategy
    # Should be adapted to be model agnostic
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
            trainable_params += [
                {
                    "params": [p for n, p in classifier.named_parameters() if n == name and p.requires_grad
                               ],
                    "lr": lr,
                }
            ]
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
            self.latent_dict[self.curr_buff] = [
                x.cpu().numpy(), y.cpu().numpy()]
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
                while rand_item_ix == 0:
                    rand_item_ix = random.choice(max_class_list)
                max_class_list.remove(rand_item_ix)
                self.latent_dict.pop(rand_item_ix)
                self.rehearsal_ixs.remove(rand_item_ix)
                self.curr_buff += 1
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
        replay_idxs = []
        k = 1
        count = 0
        budget = self.sleep_mb_size * self.sleep_n_iter
        while count < budget:
            for c in class_list:
                ixs_current_class = ixs[labels == c]
                sel_idx = np.random.choice(
                    ixs_current_class, size=k, replace=False)
                count += k
                replay_idxs.append(torch.from_numpy(sel_idx))
                if count >= budget:
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

        total_loss = 0
        params_F = self.get_layerwise_params(self.classifier_F, self.lr)
        optimizer = SGD(
            params_F, lr=self.lr)
        # optimizer = Adam(params, lr=)

        classifier_F = self.classifier_F.cuda()
        # self.classifier_G.train()
        classifier_F.train()
        total_stored_samples = len(self.rehearsal_ixs)
        print("Number of stored samples: ", total_stored_samples)
        batch = self.sleep_mb_size
        num_iter = self.sleep_n_iter
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, steps_per_epoch=int(num_iter), epochs=1
        )

        replay_ids = self.sample_memory()
        total_stored_samples = len(replay_ids)

        # Get the shape of latent activations
        replay_shape = list(self.latent_dict[0][0].shape)
        shape = [len(replay_ids)]
        shape.extend(replay_shape)
        shape = tuple(shape)

        features = np.empty((shape), dtype=np.float32)
        labels = torch.empty((len(replay_ids)), dtype=torch.long).cuda()
        for ii, v in enumerate(replay_ids):
            v = v.item()
            features[ii] = self.latent_dict[v][0]
            labels[ii] = torch.from_numpy(self.latent_dict[v][1])

        for i in range(num_iter):
            start = i * batch
            if start > (total_stored_samples - batch):
                end = total_stored_samples
            else:
                end = (i + 1) * batch
            feat_batch = torch.from_numpy(features[start:end])
            assert (
                feat_batch.nelement() != 0
            ), f"Batch is empty at start: {start}, end: {end}"
            labels_batch = labels[start:end].cuda()
            output = classifier_F(feat_batch.cuda())

            loss = self.sleep_criterion(output, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            total_loss = loss.item() + i * total_loss / (i + 1)

            if (i + 1) % 5000 == 0 or i == 0 or (i + 1) == num_iter:
                print("Iter:", (i + 1), "-- Loss: %1.5f" % (total_loss/(i+1)))

    def online_update(self, z, label, recent_labels):
        label = label.squeeze()
        for x, y in zip(z, label):
            self.aol.fit(
                x,
                y.view(
                    1,
                ),
            )

    def _update_clasifier(self, recent_class_list):
        bias = torch.ones(1).cuda()
        state_dict = self.classifier_F.state_dict().keys()
        weight_key = [k for k in state_dict if "weight" in k]
        bias_key = [k for k in state_dict if "bias" in k]
        for k in recent_class_list:
            k = torch.tensor(k, dtype=torch.int32)
            mu_k = self.aol.grab_mean(k)
            self.classifier_F.state_dict(
            )[weight_key[-1]][k] = mu_k
            self.classifier_F.state_dict()[bias_key[-1]][k] = bias

    def forward(self, sleep=False):
        """Compute the model's output given the current mini-batch."""

        with torch.no_grad():
            self.model.eval()
            self.classifier_G.eval()
            self.classifier_F.eval()
            feat = self.classifier_G(self.mb_x)
            penult_feat, output = self.classifier_F(feat, feat=True)

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

        recent_class_list = np.unique(np.concatenate(recent_labels))
        self._update_clasifier(recent_class_list)
        if self.sleep_frequency - self.rotation == 0:
            self.sleep_training(self.sleep_mb_size)
            self.rotation = 0

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
