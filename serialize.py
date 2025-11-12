import struct
from io import BytesIO, BufferedWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
import torch
def torch_dtype_to_ros_dtype(dtype: torch.dtype):
    if dtype == torch.int8:
        return 1
    if dtype == torch.int16:
        return 2
    if dtype == torch.int32:
        return 3
    if dtype == torch.int64:
        return 4
    if dtype == torch.float32:
        return 5
    if dtype == torch.float64:
        return 6
    else:
        return 0

def labeled_write(f, name, *args):
    print(f"{name}: \t {args}")
    f.write(*args)

dummy_state_dict = {
    "weight_a": torch.randn(6, 6),
    "weight_b": torch.randn(6, 6),
    "bias": torch.randn(6, 6),
    "weight_c": torch.randn(6, 6),
}

def pack_tensor_metadata(offsets: list[int], name: str, tensor: torch.Tensor, f: BufferedWriter):
    # Tensor info
    labeled_write(f, "tensor_name_len", struct.pack("<H", len(name)))
    labeled_write(f, "tensor_name_encoded", name.encode())
    labeled_write(f, "tensor_dtype", struct.pack("<B", torch_dtype_to_ros_dtype(tensor.dtype)))  # float32
    labeled_write(f, "tensor_ndims", struct.pack("<H", len(tensor.shape)))
    for dim in tensor.shape:
        labeled_write(f, "tensor_dim", struct.pack("<I", dim))     # dim0
    offsets.append(f.tell())
    labeled_write(f, "offset", struct.pack("<I", 64))

def write_tensor_data_and_patch_offsets(offsets, i, tensor, f: BufferedWriter):
    # 1. Get raw bytes of tensor data (float32)
    data_bytes = tensor.flatten().numpy().astype("<f4").tobytes()

    # 2. Record actual data offset
    actual_offset = f.tell()

    # 3. Write the tensor data
    f.write(data_bytes)

    # 4. Patch the offset in the header
    f.seek(offsets[i])
    f.write(struct.pack("<I", actual_offset))

    # 5. Seek back to end of file to continue writing next tensor
    f.seek(0, 2)   # SEEK_END


def serialize_state_dict(state_dict: dict[str, torch.Tensor], f: BufferedWriter):
    offsets = []
    labeled_write(f, "magic", b"CLLM")
    labeled_write(f, "version", struct.pack("<I", 1))     # version
    labeled_write(f, "tensor_count", struct.pack("<I", len(state_dict)))     # tensor_count
    for name, tensor in state_dict.items():
        pack_tensor_metadata(offsets, name, tensor, f)
    for idx, (_, tensor) in enumerate(state_dict.items()):
        write_tensor_data_and_patch_offsets(offsets, idx, tensor, f)

def main():
    hf_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")
    sd = hf_model.state_dict()
    for key, value in sd.items():
        print(f"{key}: {list(value.shape)}")
    with open("pythia-160m.cllm", "wb") as f:
        serialize_state_dict(sd, f)



if __name__ == "__main__":
    main()
