from deepspeed.profiling.flops_profiler.profiler import get_model_profile
from torch import nn
import argparse
import os
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small, efficientnet_b0

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

os.environ["HTTPS_PROXY"] = "http://icdvm14.ewi.tudelft.nl:3128"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    latent_layer = 8
    channel = 80
    w = 14
    h = 14
    if args.arch == 'mobilenet_v3_large' or args.arch == 'mobilenet_v3_small':
        from model_v4.mobnet import MobNet_ClassifierF, MobNet_ClassifierG
        og_model = mobilenet_v3_large()
        if args.arch == 'mobilenet_v3_small':
            og_model = mobilenet_v3_small()
            latent_layer = 5
            channel = 40
            w = 28
            h = 28

    elif args.arch == 'efficientnet_b0':
        og_model = efficientnet_b0()
        from model_v4.effnet import MobNet_ClassifierF, MobNet_ClassifierG

    else:
        raise Exception('Invalid architecture')

    class FullNet(nn.Module):
        def __init__(self, classG, classF):
            super().__init__()
            self.classG = classG
            self.classF = classF

        def forward(self, x):
            feat = self.classG(x)
            out = self.classF(feat)
            return out

    classG = MobNet_ClassifierG(latent_layer=latent_layer, arch=args.arch)
    classF = MobNet_ClassifierF(
        latent_layer=latent_layer, num_classes=100, arch=args.arch)

    fullNetwork = FullNet(classG, classF)
    file_out = f'logs/metrics_{args.arch}_{args.batch_size}'

    classG.cuda()
    classF.cuda()
    fullNetwork.cuda()

    print('\n ############## Metrics for classifier G ############## \n')

    flops, macs, params = get_model_profile(model=classG, input_shape=(
        args.batch_size, 3, 224, 224), warm_up=10, output_file=file_out+'_G.log')
    print(f'PARAMS COUNTED MANUALLY: {count_parameters(classG)}')

    print('\n ############## Metrics for classifier F ############## \n')

    flops, macs, params = get_model_profile(model=classF, input_shape=(
        args.batch_size, channel, w, h), warm_up=10, output_file=file_out+'_F.log')

    print(f'PARAMS COUNTED MANUALLY: {count_parameters(classF)}')

    print('\n ############## Metrics for entire net ############## \n')

    flops, macs, params = get_model_profile(model=fullNetwork, input_shape=(
        args.batch_size, 3, 224, 224), warm_up=10, output_file=file_out+'_all.log')

    print(f'PARAMS COUNTED MANUALLY: {count_parameters(fullNetwork)}')

    print('\n ############## Metrics for entire net ############## \n')
    flops, macs, params = get_model_profile(model=og_model, input_shape=(
        args.batch_size, 3, 224, 224), warm_up=10, output_file=file_out+'_og.log')
    print(f'PARAMS COUNTED MANUALLY: {count_parameters(og_model)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default="mobilenet_v3_small")
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    main(args)
