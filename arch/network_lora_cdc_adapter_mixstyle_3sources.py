# from .timm_vit import _create_vit_adapter
# import logging
# import timm
#from timm.models.vision_transformer import vit_base_patch16_224
#from .convpass import set_Convpass
from .timm_lora_adapter_cdc_mixstyle_3sources import vit_base_patch16_224, vit_base_patch16_224_in21k, vit_tiny_patch16_224_in21k, vit_small_patch16_224_in21k, vit_large_patch16_224_in21k
import numpy as np


def build_net(arch_name, pretrained, **kwargs):
    num_classes = kwargs['num_classes']

    depth = 12
    # model_cfg = {
    #    'super_LoRA_dim': kwargs['super_LoRA_dim'],
    #    'super_prompt_tuning_dim': kwargs['super_prompt_tuning_dim'],
    #    'super_adapter_dim': kwargs['super_adapter_dim'],
    #    'super_prefix_dim': kwargs['super_prefix_dim'],
    #    'depth':12
    #}
    if arch_name == 'vit_base_patch16_224':
        model = vit_base_patch16_224(pretrained, **kwargs) # todo
    
    if arch_name == 'vit_base_patch16_224_in21k':
        model = vit_base_patch16_224_in21k(pretrained, **kwargs) # todo

    if arch_name == 'vit_tiny_patch16_224_in21k':
        model = vit_tiny_patch16_224_in21k(pretrained, **kwargs) # todo

    if arch_name == 'vit_small_patch16_224_in21k':
        model = vit_small_patch16_224_in21k(pretrained, **kwargs) # todo

    if arch_name == 'vit_large_patch16_224_in21k':
        model = vit_large_patch16_224_in21k(pretrained, **kwargs) # todo

    set_sample_config = {
        'visual_prompt_dim': [kwargs['super_prompt_tuning_dim']] * depth,
        'lora_dim': [kwargs['super_LoRA_dim']]* depth,
        'adapter_dim': [kwargs['super_adapter_dim']] * depth,
        'prefix_dim': [kwargs['super_prefix_dim']] * depth,
    }
    # model.cuda()
    model.set_sample_config(set_sample_config)

    return model

if __name__ == '__main__':
    import torch
    kwargs = {
        # 'conv_type': self.config.MODEL.CONV,
        'num_classes': 2,
        # 'cdc_theta': self.config.MODEL.CDC_THETA,
        'super_LoRA_dim': 32,
        'super_prompt_tuning_dim': 0,
        'super_adapter_dim': 8,
        'super_prefix_dim': 0,
    }

    model = build_net(arch_name='vit_base_patch16_224_in21k', pretrained=True, **kwargs)
    model = model.to('cuda:1')
    para_num = 0
    for name, p in model.named_parameters():
        if 'adapter' not in name and 'prompt' not in name and 'LoRA' not in name and 'prefix' not in name and 'head' not in name:
            p.requires_grad = False
        else:
            p.requires_grad = True
            para_num += np.prod(p.size())
        print(name, p.requires_grad, p.shape)
    print('Trainable para number:', para_num)
    x=torch.rand(2,3,224,224).to('cuda:1')
    y, feat=model(x)
    print(y.size(), feat.size())
    # import IPython; IPython.embed()

