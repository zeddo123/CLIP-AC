import torch
from PIL import Image
import open_clip

from compositional_prompt_ensemble import build_prompt_ens


class ClipAC:
    def __init__(self, label,
                vit_model='ViT-B-32',
                pretrained='laion2b_s34b_b79k',
                device: str = 'cuda'):
        self.device = device
        self.label = label
        self.normal_prompts, self.anomaly_prompts = build_prompt_ens(self.label)
        self.prompts = self.normal_prompts + self.anomaly_prompts 

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(vit_model,
                pretrained=pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(vit_model)

        self.text = tokenizer(self.prompts).to(device)

    def __call__(self, image):
        img = self.preprocess(Image.open(image)).unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(img)
            text_features = self.model.encode_text(self.text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        return text_probs[:, len(self.normal_prompts):].sum().cpu().numpy()
