import torch

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs


def simple_inference(pairs, model, use_amp=False, feature_only=True):
    view1, view2 = pairs[0]
    for key in ['true_shape', 'idx', 'instance']:
        view1.pop(key)
        view2.pop(key)

    view1['img'] = view1['img'].to('cuda', non_blocking=True)
    view2['img'] = view2['img'].to('cuda', non_blocking=True)

    # print(view1.keys())
    # exit(1)

    with torch.cuda.amp.autocast(enabled=use_amp):
        pred1, pred2 = model(view1, view2, feature_only=feature_only)

    return pred1, pred2


if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    # size = 224
    size = 512
    # decoder_arch = 'linear'
    decoder_arch = 'dpt'

    model_name = f"naver/DUSt3R_ViTLarge_BaseDecoder_{size}_{decoder_arch}"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    images = load_images(['../../datasets/origin_nyuv2_plane/0_d2.npz', '../../datasets/origin_nyuv2_plane/0_d2.npz'], size=256)
    # pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=False)
    pairs = [(images[0], images[1])]

    # print(type(pairs[0][0]['img']))
    # exit(1)

    # output = inference(pairs, model, device, batch_size=batch_size, feature_only=True)

    pred1, pred2 = simple_inference(pairs, model, feature_only=True)
    print(type(pred1))
    print(pred1.keys())

    for feat in pred1['feat']:
        print(feat.size())

    exit(1)

    # at this stage, you have the raw dust3r predictions
    # view1, pred1 = output['view1'], output['pred1']
    # view2, pred2 = output['view2'], output['pred2']
