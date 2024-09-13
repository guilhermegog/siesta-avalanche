from model_v2 import SplitModel, mobilenet_v3_large
import torch


def load_partial(model,  checkpoint, part="all"):
    state_dict = checkpoint["state_dict"]
    if part == 'h_net':
        partial_dict = {k: v for k, v in state_dict.items() if k.startswith(
            'features.') and int(k.split('.')[1]) < 8}
    elif part == 'g_net':
        partial_dict = {k: v for k, v in state_dict.items() if k.startswith(
            'features.') and int(k.split('.')[1]) >= 8}
    elif part == 'f_net':
        partial_dict = {k: v for k, v in state_dict.items()
                        if k.startswith('classifier')}
    elif part == 'all':
        partial_dict = state_dict
    else:
        raise ValueError(
            "Invalid part specified. Choose' h_net', 'g_net', 'f_net', or 'all'.")

    model.load_state_dict(partial_dict, strict=False)
    return model


def build_model(split_layer=8):
    model = mobilenet_v3_large()
    checkpoint = torch.load(
        "checkpoints/best_cosine_softmax_loss_SWAV_sgd_layerlr02_step_MIXUP_CUTMIX_50e_100c.pth")

    split_model = SplitModel(model, split_layer=8)

    load_partial(model=split_model, checkpoint=checkpoint, part="h_net")
    return split_model
