import torch
import pickle
from torch import nn
from model_v2.siesta_mobilenet import (mobilenet_v3_large, CosineLinear,
                                       LargeMobileNet, MobNetClassifyAfterLayer8,
                                       MobNet_StartAt_Layer8)

class CMA(object):
    """
    A continual moving average for tracking loss updates.
    """
    def __init__(self):
        self.N = 0
        self.avg = 0.0
    def update(self, X):
        self.avg = (X + self.N * self.avg) / (self.N + 1)
        self.N = self.N + 1
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


def build_classifier(classifier, classifier_ckpt, num_classes):  # for swav
    classifier = eval(classifier)(num_classes=num_classes)

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


class RunningMean(nn.Module):

    def __init__(self, input_shape, num_classes, device='cuda'):

        super(RunningMean, self).__init__()

        # parameters
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes

        # initialize
        self.muK = torch.zeros((num_classes, input_shape),
                               dtype=torch.float64).to(self.device)
        self.cK = torch.zeros(num_classes).to(self.device)

    @torch.no_grad()
    def fit(self, x, y):
        x = x.to(self.device)
        y = y.long().to(self.device)

        # make sure things are the right shape
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)

        # update class means
        self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
        self.cK[y] += 1

    def grab_mean(self, y):
        return self.muK[y]


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


def create_classifiers():
    """
    Creates mobilenet as described in the SIESTA paper
    and as implemented in the original source code: 
    https://github.com/yousuf907/SIESTA/tree/main
    """
    ckpt = "/space/gguedes/Documents/sanity_check/checkpoints/swav_100c_2000e_mobilenet_modified_gelu_updated.pth"
    classifier_F = build_classifier(
        "MobNet_StartAt_Layer8", ckpt, num_classes=1000)
    core_model = build_classifier(
        "MobNetClassifyAfterLayer8", ckpt, num_classes=1000)
    classifier_G = ModelWrapper(core_model, output_layer_names=[
                                "model.features.7"], return_single=True)

    old_state = torch.load(
        "checkpoints/best_cosine_softmax_loss_SWAV_sgd_layerlr02_step_MIXUP_CUTMIX_50e_100c.pth")
    safe_load_dict(
        classifier_F, old_state['state_dict'], should_resume_all_params=False)
    old_state = old_state["state_dict"]
    new_state = classifier_F.state_dict()
    for k in old_state:
        assert torch.equal(old_state[k].cpu(),
                           new_state[k[len("module."):]]), k
    print("Successfully performed sanity check!!")

    aol = RunningMean(1280, 1000)  # Awake online learning

    return classifier_G, classifier_F, aol


def get_buffer():
    with open("checkpoints/swav_buffer_100c.pkl", "rb") as file:
        buffer = pickle.load(file)
    return buffer
