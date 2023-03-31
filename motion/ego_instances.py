import torch
from tqdm import tqdm

from OGC.models.segnet_kitti import MaskFormer3D
from my_datasets.argoverse.argoverse2 import Argoverse2_Sequence

sequence = Argoverse2_Sequence(sequence_nbr=0)

segnet = MaskFormer3D(n_slot=255,    # n_slot means number of output instances
                          use_xyz=True,
                          n_transformer_layer=2,
                          transformer_embed_dim=128,
                          transformer_input_pos_enc=False).cuda()

data_dict = sequence.get_frame(40)


for i in tqdm(range(20)):
    pts = torch.tensor(data_dict['pts'][:,:3], dtype=torch.float).unsqueeze(0).cuda().contiguous()
    point_feats = pts.clone().cuda().contiguous()
    mask = segnet(pts[...,:3], point_feats[...,:3])

    print(torch.cuda.mem_get_info(0)[0] / 1024 ** 2, 'Mb')

