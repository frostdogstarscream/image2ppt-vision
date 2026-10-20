"""Real CUDA FP32 and actual local model initialization; no network or fixtures."""
import json
import torch
from image2ppt_vision.models import capabilities, sam_model, model_root, ocr_engine

assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count()==1, 'expected only the assigned GPU'
model=sam_model(str(model_root()))
assert next(model.parameters()).dtype==torch.float32
x=torch.ones((256,256),device='cuda',dtype=torch.float32)
assert float((x@x)[0,0])==256
ocr_engine(str(model_root()))
print(json.dumps({'status':'PASS','device':torch.cuda.get_device_name(0),
                  'visible_devices':torch.cuda.device_count(),'model_dtype':str(next(model.parameters()).dtype),
                  'allocated_bytes':torch.cuda.memory_allocated(0),'capabilities':capabilities()}))
