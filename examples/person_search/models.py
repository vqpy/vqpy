import os.path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

__fast_reid_repository__ = "JDAI-CV/fast-reid:v1.3.0"

ROOT_URL = "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/"

__weights_urls__ = {
    "MSMT17/bagtricks_S50.yml": ROOT_URL + "msmt_bot_S50.pth",
    "MSMT17/bagtricks_R50.yml": ROOT_URL + "msmt_bot_R50.pth",
    "MSMT17/bagtricks_R50-ibn.yml": ROOT_URL + "msmt_bot_R50-ibn.pth",
    "MSMT17/bagtricks_R101-ibn.yml": ROOT_URL + "msmt_bot_R101-ibn.pth",
    "MSMT17/AGW_S50.yml": ROOT_URL + "msmt_agw_S50.pth",
    "MSMT17/AGW_R50.yml": ROOT_URL + "msmt_agw_R50.pth",
    "MSMT17/AGW_R50-ibn.yml": ROOT_URL + "msmt_agw_R50-ibn.pth",
    "MSMT17/AGW_R101-ibn.yml": ROOT_URL + "msmt_agw_R101-ibn.pth",
    "MSMT17/sbs_S50.yml": ROOT_URL + "msmt_sbs_S50.pth",
    "MSMT17/sbs_R50.yml": ROOT_URL + "msmt_sbs_R50.pth",
    "MSMT17/sbs_R50-ibn.yml": ROOT_URL + "msmt_sbs_R50-ibn.pth",
    "MSMT17/sbs_R101-ibn.yml": ROOT_URL + "msmt_sbs_R101-ibn.pth",
    "MSMT17/mgn_R50-ibn.yml": ROOT_URL + "msmt_mgn_R50-ibn.pth",

    "Market1501/bagtricks_S50.yml": ROOT_URL + "market_bot_S50.pth",
    "Market1501/bagtricks_R50.yml": ROOT_URL + "market_bot_R50.pth",
    "Market1501/bagtricks_R50-ibn.yml": ROOT_URL + "market_bot_R50-ibn.pth",
    "Market1501/bagtricks_R101-ibn.yml": ROOT_URL + "market_bot_R101-ibn.pth",
    "Market1501/AGW_S50.yml": ROOT_URL + "market_agw_S50.pth",
    "Market1501/AGW_R50.yml": ROOT_URL + "market_agw_R50.pth",
    "Market1501/AGW_R50-ibn.yml": ROOT_URL + "market_agw_R50-ibn.pth",
    "Market1501/AGW_R101-ibn.yml": ROOT_URL + "market_agw_R101-ibn.pth",
    "Market1501/sbs_S50.yml": ROOT_URL + "market_sbs_S50.pth",
    "Market1501/sbs_R50.yml": ROOT_URL + "market_sbs_R50.pth",
    "Market1501/sbs_R50-ibn.yml": ROOT_URL + "market_sbs_R50-ibn.pth",
    "Market1501/sbs_R101-ibn.yml": ROOT_URL + "market_sbs_R101-ibn.pth",
    "Market1501/mgn_R50-ibn.yml": ROOT_URL + "market_mgn_R50-ibn.pth",

    "DukeMTMC/bagtricks_S50.yml": ROOT_URL + "duke_bot_S50.pth",
    "DukeMTMC/bagtricks_R50.yml": ROOT_URL + "duke_bot_R50.pth",
    "DukeMTMC/bagtricks_R50-ibn.yml": ROOT_URL + "duke_bot_R50-ibn.pth",
    "DukeMTMC/bagtricks_R101-ibn.yml": ROOT_URL + "duke_bot_R101-ibn.pth",
    "DukeMTMC/AGW_S50.yml": ROOT_URL + "duke_agw_S50.pth",
    "DukeMTMC/AGW_R50.yml": ROOT_URL + "duke_agw_R50.pth",
    "DukeMTMC/AGW_R50-ibn.yml": ROOT_URL + "duke_agw_R50-ibn.pth",
    "DukeMTMC/AGW_R101-ibn.yml": ROOT_URL + "duke_agw_R101-ibn.pth",
    "DukeMTMC/sbs_S50.yml": ROOT_URL + "duke_sbs_S50.pth",
    "DukeMTMC/sbs_R50.yml": ROOT_URL + "duke_sbs_R50.pth",
    "DukeMTMC/sbs_R50-ibn.yml": ROOT_URL + "duke_sbs_R50-ibn.pth",
    "DukeMTMC/sbs_R101-ibn.yml": ROOT_URL + "duke_sbs_R101-ibn.pth",
    "DukeMTMC/mgn_R50-ibn.yml": ROOT_URL + "duke_mgn_R50-ibn.pth",
}


def _build_model(cfg, pretrained=True, verbose=True):
    from torch.hub import (
        _get_cache_or_reload,
        load_state_dict_from_url
    )

    import importlib.util
    import inspect

    if 'trust_repo' in inspect.getfullargspec(_get_cache_or_reload):  #
        repo_dir = _get_cache_or_reload(
            __fast_reid_repository__,
            force_reload=False, verbose=verbose,
            trust_repo="check", calling_fn="load")
    else:
        repo_dir = _get_cache_or_reload(
            __fast_reid_repository__,
            force_reload=False, verbose=verbose)

    load_state_dict_from_url(__weights_urls__[cfg], progress=verbose)

    # import pip
    # pip.main(["install", "-t", repo_dir, "-r",
    #  repo_dir + "/docs/requirements.txt"])

    sys.path.insert(0, repo_dir)

    assert importlib.util.find_spec('fastreid.config.config'), \
        "could not import Fast-ReID module: fastreid.config.config"
    _cfg = importlib.import_module('fastreid.config.config').get_cfg()
    _cfg.merge_from_file(os.path.join(repo_dir, 'configs', cfg))
    _cfg.MODEL.BACKBONE.PRETRAIN = False

    assert importlib.util.find_spec('fastreid.modeling.meta_arch.build'), \
        "could not import Fast-ReID module: fastreid.modeling.meta_arch.build"
    build = importlib.import_module('fastreid.modeling.meta_arch.build')
    model = build.build_model(_cfg)
    model.eval()

    if pretrained:
        assert importlib.util.find_spec('fastreid.utils.checkpoint'), \
            "could not import Fast-ReID module: fastreid.utils.checkpoint"
        checkpoint = importlib.import_module('fastreid.utils.checkpoint')
        checkpointer = checkpoint.Checkpointer
        checkpointer(model).load(path=os.path.join(
            os.path.join(torch.hub.get_dir(), 'checkpoints'),
            os.path.split(__weights_urls__[cfg])[1]))

    sys.path.remove(repo_dir)

    return model


class ReIDPredictor(object):

    def __init__(self, *, cfg, pretrained=True, verbose=True):
        super(ReIDPredictor, self).__init__()

        self.cfg = cfg
        self.pretrained = pretrained
        self.verbose = verbose

        if cfg not in __weights_urls__.keys():
            raise NotImplementedError

        self.model = _build_model(cfg, pretrained, verbose)

    def __call__(self, image):
        if isinstance(image, str):
            image = to_tensor(Image.open(image))
        elif isinstance(image, np.ndarray):
            image = to_tensor(image)

        if len(image.shape) == 3:
            image = image.unsqueeze(dim=0)

        inputs = {"images": image.to(self.model.device)}

        with torch.no_grad():
            predictions = self.model(inputs)
        predictions = F.normalize(predictions, dim=1, p=2)

        return predictions.detach().cpu().numpy()
