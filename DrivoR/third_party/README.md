# Visual backbone sources

## DINOv3

`dinov3/` is a local checkout of the official [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) repository, used by the top-level DrivoR Action-DiT DINOv3 image adapter. Its license and notices remain in [dinov3/LICENSE.md](dinov3/LICENSE.md).

The adapter imports only the backbone package when a DINOv3 configuration is selected. Existing DINOv2 paths continue using the installed timm version. Backbone implementation files are unmodified; Q/V LoRA and optional Scene-query handling live in `navsim/agents/drivoR_action_dit/dinov3_image_encoder.py`.

Downloaded ModelScope weights and their original metadata/license files are under `../weights/dinov3/vits16/` and `../weights/dinov3/vitl16/`. They are not stored inside this source checkout.

## V-JEPA 2

`vjepa2/` is a local checkout of the official [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) repository. Its backbone implementation is unmodified, and its license and notices remain in [vjepa2/LICENSE](vjepa2/LICENSE) and the upstream source files.

The Action-DiT adapter imports the native ViT-L only when a V-JEPA 2 configuration is selected. Single-image tubelet adaptation, Q/V LoRA and optional Scene queries live in `navsim/agents/drivoR_action_dit/vjepa2_image_encoder.py`. Weights from the official README are under `../weights/vjepa2/vitl/`; training loads the extracted `target_encoder.safetensors`.

V-JEPA 2.1 uses the same checkout's `app/vjepa_2_1/models/` package through `navsim/agents/drivoR_action_dit/vjepa2_1_image_encoder.py`. Its pretrained native image stem, image modality embedding and scaled RoPE are retained. The official ViT-L weights are under `../weights/vjepa2_1/vitl/`; this variant loads `ema_encoder.safetensors`.
