import os
import torch
from sentence_transformers import SentenceTransformer
from utils.labels_dict import UNI_UID2UNAME, UNAME2EM_NAME

UNI_UNAME2ID = {v: i for i, v in UNI_UID2UNAME.items()}


def create_embs_from_names(labels, other_descriptions=None):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CLIP_TEXT_MODEL = SentenceTransformer('clip-ViT-B-32', device=DEVICE)
    u_descrip_dir = 'data/clip_descriptions'
    embs = []
    for name in labels:
        if name in UNAME2EM_NAME.keys():
            with open(os.path.join(u_descrip_dir, UNAME2EM_NAME[name] + '.txt'), 'r') as f:
                description = f.readlines()[0]
        elif name in other_descriptions:
            description = other_descriptions[name]

        with torch.no_grad():
            text_features = CLIP_TEXT_MODEL.encode([description, ], convert_to_tensor=True, device=DEVICE)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        embs.append(text_features)
    embs = torch.stack(embs, dim=0).squeeze()
    return embs
