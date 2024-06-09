import torch

def get_backbone_weights_from_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    old_state_dict = checkpoint['state_dict']
    filtered_state_dict = {k: v for k, v in old_state_dict.items() if k.startswith('model.backbone')}
    filtered_state_dict = {k.replace('model.backbone.', 'backbone.'): v for k, v in filtered_state_dict.items()}
 
    return filtered_state_dict