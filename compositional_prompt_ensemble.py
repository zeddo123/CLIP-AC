state_level_normal_prompts = [
    '[o]',
    'flawless [o]',
    'perfect [o]',
    'unblemished [o]',
    '[o] without flaw',
    '[o] without defect',
    '[o] without damage',
]

state_level_anomaly_prompts = [
    'damaged [o]',
    '[o] with flaw',
    '[o] with defect',
    '[o] with damage'
]

template_level_prompts = [
    'a cropped photo of the [c]',
    'a cropped photo of a [c]',
    'a close-up photo of a [c]',
    'a close-up photo of the [c]',
    'a bright photo of a [c]',
    'a bright photo of the [c]',
    'a dark photo of the [c]',
    'a dark photo of a [c]',
    'a jpeg corrupted photo of a [c]',
    'a jpeg corrupted photo of the [c]',
    'a blurry photo of the [c]',
    'a blurry photo of a [c]',
    'a photo of a [c]',
    'a photo of the [c]',
    'a photo of a small [c]',
    'a photo of the small [c]',
    'a photo of a large [c]',
    'a photo of the large [c]',
    'a photo of the [c], for visual inspection',
    'a photo of a [c], for visual inspection',
    'a photo of the [c], for anomaly detection',
    'a photo of a [c], for anomaly detection'
]

def build_prompt_ens(label: str):
    normal_prompts = []
    anomaly_prompts = []

    for prompt in state_level_normal_prompts:
        state_level_prompt = prompt.replace('[o]', label)
        for template in template_level_prompts:
            final_prompt = template.replace('[c]', state_level_prompt)
            normal_prompts.append(final_prompt)

    for prompt in state_level_anomaly_prompts:
        state_level_prompt = prompt.replace('[o]', label)
        for template in template_level_prompts:
            final_prompt = template.replace('[c]', state_level_prompt)
            anomaly_prompts.append(final_prompt)

    return normal_prompts, anomaly_prompts
