# -*- coding: utf-8 -*-
from . import DFFN, SSRN, MFERN, pResNet, HybridSN, RSSAN, SSTN, ASKResNet, ResNet, DCRN
    # , RSSAN, SSTN, SPRLT

# get models by name
def get_model(name, dataset, n_classes, n_bands, patch_size):
    if name == 'SSRN':
        model = SSRN.SSRN(n_bands, n_classes)
    elif name == 'DFFN':
        model = DFFN.DFFN(dataset, n_bands, n_classes)
    elif name == 'MFERN':
        model = MFERN.MFERN(dataset, n_bands, n_classes)

    # 对比模型
    elif name == 'ResNet':
        model = ResNet.ResNet34(n_bands, n_classes)
    elif name == 'pResNet':
        model = pResNet.PResNet(n_bands, n_classes)
    elif name == 'ASKResNet':
        model = ASKResNet.ASKResNet(n_bands, n_classes)
    elif name == 'HybridSN':
        model = HybridSN.HybridSN(dataset, n_bands, n_classes)
    elif name == 'RSSAN':
        model = RSSAN.RSSAN(n_bands, n_classes)
    elif name == 'SSTN':
        model = SSTN.SSTN(dataset, n_bands, n_classes)
    elif name == 'DCRN':
        model = DCRN.proposed(dataset, n_bands, n_classes)

    else:
        raise KeyError("{} model is unknown.".format(name))

    return model