"""LPRNet Model"""

# Provided interface:
# GetLP: infer the license plate from the image of a car

import torch
import numpy as np
import cv2

device = torch.device("cuda")

CHARS_ASCII = ['BJ-', 'SH-', 'TJ-', 'CQ-', 'HE-', 'SX-', 'IM-', 'LN-', 'JL-',
               'HL-', 'JS-', 'ZJ-', 'AH-', 'FJ-', 'JX-', 'SD-', 'HA-', 'HB-',
               'HN-', 'GZ-', 'GX-', 'HI-', 'SC-', 'GZ-', 'YN-', 'XZ-', 'SX-',
               'GS-', 'QH-', 'NX-', 'XJ-',
               '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
               'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
               'W', 'X', 'Y', 'Z', 'I', 'O', '-'
               ]

pnet, onet, lprnet, stnet, mini_lp = None, None, None, None, None


def network_setup():
    import sys

    sys.path.append("./lpdetect")
    sys.path.append("./lpdetect/LPRNet")
    sys.path.append("./lpdetect/MTCNN")

    from lpdetect.LPRNet.model.LPRNET import LPRNet, CHARS
    from lpdetect.LPRNet.model.STN import STNet
    from lpdetect.MTCNN.MTCNN import PNet, ONet

    global pnet, onet, lprnet, stnet, mini_lp
    pnet = PNet().to(device)
    pnet.load_state_dict(
        torch.load('lpdetect/MTCNN/weights/pnet_Weights',
                   map_location=lambda storage, loc: storage)
        )
    pnet.eval()
    onet = ONet().to(device)
    onet.load_state_dict(
        torch.load('lpdetect/MTCNN/weights/onet_Weights',
                   map_location=lambda storage, loc: storage)
        )
    onet.eval()
    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0).to(device)
    lprnet.load_state_dict(
        torch.load('lpdetect/LPRNet/weights/Final_LPRNet_model.pth',
                   map_location=lambda storage, loc: storage)
        )
    lprnet.eval()
    stnet = STNet().to(device)
    stnet.load_state_dict(
        torch.load('lpdetect/LPRNet/weights/Final_STN_model.pth',
                   map_location=lambda storage, loc: storage))
    stnet.eval()
    mini_lp = (50, 15)  # smallest lp size


def GetLP(image):
    try:
        from lpdetect.LPRNet.LPRNet_Test import decode as lprnet_decode
        from lpdetect.MTCNN.MTCNN import detect_pnet, detect_onet
    except ImportError:
        raise ImportError("Please run VQPY/download_models.sh before using \
         built-in vehicle properties.")

    from vqpy.utils import crop_image

    if device is None:
        network_setup()
    if image is None:
        return None
    bboxes = detect_pnet(pnet, image, mini_lp, device)
    bboxes = detect_onet(onet, image, bboxes, device)
    if len(bboxes) == 0:
        return None
    bboxes = bboxes[np.argsort(-bboxes[:, 4])]
    for i in range(len(bboxes)):
        bbox = bboxes[i, :4]
        img_box = crop_image(image, bbox)
        if img_box is None:
            continue
        im = cv2.resize(img_box, (94, 24), interpolation=cv2.INTER_CUBIC)
        im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5)*0.0078125
        # data.size is torch.Size([1, 3, 24, 94])
        data = torch.from_numpy(im).float().unsqueeze(0).to(device)
        transfer = stnet(data)
        preds = lprnet(transfer)
        preds = preds.cpu().detach().numpy()  # (1, 68, 18)
        labels, _ = lprnet_decode(preds, CHARS_ASCII)
        labels = labels[0]
        if len(labels) < 7:
            continue
        return labels

    return None
