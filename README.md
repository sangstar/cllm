# `cllm`
A work-in-progress inference engine and API server implemented in pure C. `serialize.py` saves models in 
`.cllm` format to be loaded by the program for inference.

To-do list:

- [x] Serializing and deserializing weights in to `cllm_data`
- [x] Loading `cllm_data` in to a model resource
- [x] SGEMM on tensors
- [x] Dense layer forward
- [ ] Layernorm layer forward
- [ ] Embedding layer forward
- [ ] MLP layer forward
- [ ] Attention layer
- [ ] RoPE implementation
- [ ] GEMM on other data types
- [ ] Transformer block type
- [ ] Full, correct model forward pass
- [ ] Sampling
- [ ] Tokenization considerations
- [ ] API server
