# from timm import (

# )
import timm

timm_efficientnet_b0 = timm.create_model("efficientnet_b0", num_classes=1000)
# timm_efficientnet_b0=''
timm_deit_s = timm.create_model("deit_small_patch16_224", pretrained=False)
timm_deit_b = timm.create_model("deit_base_patch16_224", pretrained=False)
# timm_deit_s=''
# timm_deit_b=''
timm_vit_t = timm.create_model("vit_tiny_patch16_224", pretrained=False)
# timm_vit_t=''
