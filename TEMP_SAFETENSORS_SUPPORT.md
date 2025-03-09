# Implementing SafeTensors Support

## Key Changes Required

1. **Add safetensors to dependencies**
   - Add `safetensors` to requirements.txt
   - Recommended version: 0.3.0 or later

2. **Modify core checkpoint loading in `ldm/models/diffusion/ddpm.py`**
   - Update `init_from_ckpt` method (~line 195)
   - Detect file format based on extension
   - For .safetensors files, use `safetensors.safe_open` instead of `torch.load`
   - Example:
     ```python
     def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
         if path.endswith('.safetensors'):
             from safetensors.torch import load_file
             sd = load_file(path)
         else:
             sd = torch.load(path, map_location="cpu")
             if "state_dict" in list(sd.keys()):
                 sd = sd["state_dict"]
     ```

3. **Update `load_model_from_config` in `ldm/util.py`**
   - Modify to handle both .ckpt and .safetensors formats (~line 76)
   - Add conditional loading based on file extension
   - Update error handling to be format-aware
   - Example:
     ```python
     def load_model_from_config(config, ckpt, verbose=False):
         print(f"Loading model from {ckpt}")
         
         if ckpt.endswith('.safetensors'):
             from safetensors.torch import load_file
             pl_sd = load_file(ckpt)
             # safetensors files don't use state_dict key
             sd = pl_sd
         else:
             pl_sd = torch.load(ckpt, map_location="cpu")
             if "state_dict" in pl_sd:
                 sd = pl_sd["state_dict"]
             else:
                 sd = pl_sd
     ```

4. **Handle checkpoint saving in `dreambooth_helpers/callback_helpers.py`**
   - Add option to save in .safetensors format
   - Extend `SetupCallback` to support safetensors format
   - Implement format selection through configuration
   - Example implementation:
     ```python
     def save_checkpoint(self, trainer, save_path, save_as_safetensors=False):
         if save_as_safetensors:
             from safetensors.torch import save_file
             state_dict = trainer.model.state_dict()
             save_file(state_dict, save_path)
         else:
             trainer.save_checkpoint(save_path)
     ```

5. **Update configuration options**
   - Add a parameter to `JoePennaDreamboothConfigSchemaV1` for checkpoint format
   - Default to original format for backward compatibility
   - Example:
     ```python
     # In dreambooth_helpers/joepenna_dreambooth_config.py
     class JoePennaDreamboothConfigSchemaV1:
         def __init__(self, ..., checkpoint_format="ckpt", ...):
             self.checkpoint_format = checkpoint_format  # "ckpt" or "safetensors"
     ```
   - Update argument parser in `dreambooth_helpers/arguments.py` to accept format flag

6. **Testing with existing .safetensors models**
   - Test loading models from popular repositories that provide .safetensors
   - Verify that generated outputs match those from .ckpt equivalents
   - Test saving and reloading in .safetensors format
   - Verify no regressions in training workflow

## Additional Implementation Notes

- SafeTensors does not store Python objects, only tensors, ensuring security
- SafeTensors has faster loading times, especially for large models
- When switching to .safetensors, ensure metadata is preserved if needed
- Error messages should be clear if attempting to load malformed files
- For existing models, convert from .ckpt to .safetensors format with a utility script

## README.md Updates Needed

The README.md file needs to be updated to document the new `checkpoint_format` parameter. Add it to both:

1. The "Example Configuration File" section (around line 184):
```json
{
    "checkpoint_format": "safetensors",  // Add this line - can be "ckpt" or "safetensors"
    "class_word": "woman",
    ...
}
```

2. The "Command Line Parameters" table (around line 260):
```
| `--checkpoint_format` | string | `"safetensors"` | *Optional* Defaults to `"ckpt"`. Format to use for saved model checkpoints. Can be "ckpt" or "safetensors". |
```