import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torch.nn import Parameter
import math
from torch.nn import functional as F
from torch.nn import init
from typing import Any, Callable, Dict, Optional



class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # C x d i.e., 1000 x 1280
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, 0, 0.01)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  # for initializaiton of sigma
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.bias is not None:
            input = torch.cat(
                (input, (torch.ones(len(input), 1).cuda())), dim=1)
            concat_weight = torch.cat((self.weight, self.bias), dim=1)
            out = F.linear(F.normalize(input, p=2, dim=1, eps=1e-8),
                           F.normalize(concat_weight, p=2, dim=1, eps=1e-8))
        else:
            out = F.linear(F.normalize(input, p=2, dim=1, eps=1e-8),
                           F.normalize(self.weight, p=2, dim=1, eps=1e-8))

        if self.sigma is not None:
            out = self.sigma * out
        return out


class MobNet_ClassifierF(nn.Module):
    def __init__(self, latent_layer=8, num_classes=None):
        super(MobNet_ClassifierF, self).__init__()

        self.model = mobilenet_v3_small(weights='IMAGENET1K_V1')

        for _ in range(0, latent_layer):
            del self.model.features[0]

        if num_classes is not None:
            print('Changing output layer to contain %d classes. ' % num_classes)
            self.model.classifier[3] = CosineLinear(1024, num_classes)

    def forward(self, x, feat=False):
        out = self.model.features(x)
        out = self.model.avgpool(out)
        out = torch.flatten(out, 1)

        if feat:
            features = self.model.classifier[0](out)
            out = self.model.classifier(out)
            return features, out

        out = self.model.classifier(out)
        return out


class MobNet_ClassifierG_net (nn.Module):
    def __init__(self, latent_layer=8, num_del=0, num_classes=None):
        super(MobNet_ClassifierG_net, self).__init__()

        self.model = mobilenet_v3_small(weights='IMAGENET1K_V1')

        for _ in range(0, num_del):
            del self.model.features[latent_layer][-1]

        if num_classes is not None:
            print('Changing output layer to contain %d classes. ' % num_classes)
            self.model.classifier[3] = CosineLinear(1024, num_classes)

    def forward(self, x):
        out = self.model(x)
        return out


class MobNet_ClassifierG(MobNet_ClassifierG_net):
    def __init__(self, latent_layer=8, num_classes=None):
        super(MobNet_ClassifierG, self).__init__(latent_layer=latent_layer,
                                                 num_del=0, num_classes=num_classes)


def safe_load_dict(model, new_model_state, should_resume_all_params=False):
    old_model_state = model.state_dict()
    c = 0
    if should_resume_all_params:
        for old_name, old_param in old_model_state.items():
            assert old_name in list(new_model_state.keys()), "{} parameter is not present in resumed checkpoint".format(
                old_name)
    for name, param in new_model_state.items():
        n = name.split('.')
        beg = n[0]
        end = n[1:]
        if beg == 'module':
            name = '.'.join(end)
        if name not in old_model_state:
            #print('%s not found in old model.' % name)
            continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        c += 1
        if old_model_state[name].shape != param.shape:
            print('Shape mismatch...ignoring %s' % name)
            continue
        else:
            old_model_state[name].copy_(param)
    if c == 0:
        raise AssertionError(
            'No previous ckpt names matched and the ckpt was not loaded properly.')


def build_classifier(classifier, classifier_ckpt, latent_layer, num_classes):  # for swav
    classifier = eval(classifier)(
        latent_layer=latent_layer, num_classes=num_classes)

    if classifier_ckpt is None:
        print("Will not resume any checkpoints!")
    else:
        resumed = torch.load(classifier_ckpt)
        if 'state_dict' in resumed:
            state_dict_key = 'state_dict'
        else:
            state_dict_key = 'model_state'
        print("Resuming with {}".format(classifier_ckpt))
        safe_load_dict(
            classifier, resumed[state_dict_key], should_resume_all_params=False)
    return classifier


def get_name_to_module(model):
    name_to_module = {}
    for m in model.named_modules():
        name_to_module[m[0]] = m[1]
    return name_to_module


def get_activation(all_outputs, name):
    def hook(model, input, output):
        all_outputs[name] = output.detach()

    return hook


def add_hooks(model, outputs, output_layer_names):
    """
    :param model:
    :param outputs: Outputs from layers specified in `output_layer_names` will be stored in `output` variable
    :param output_layer_names:
    :return:
    """
    name_to_module = get_name_to_module(model)
    for output_layer_name in output_layer_names:
        name_to_module[output_layer_name].register_forward_hook(
            get_activation(outputs, output_layer_name))


class ModelWrapper(nn.Module):
    def __init__(self, model, output_layer_names, return_single=False):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.output_layer_names = output_layer_names
        self.outputs = {}
        self.return_single = return_single
        add_hooks(self.model, self.outputs, self.output_layer_names)

    def forward(self, images):
        self.model(images)
        output_vals = [self.outputs[output_layer_name]
                       for output_layer_name in self.output_layer_names]
        if self.return_single:
            return output_vals[0]
        else:
            return output_vals
