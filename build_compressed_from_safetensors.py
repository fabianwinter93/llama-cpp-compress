import os
import sys
from pathlib import Path

import argparse

import re

import torch
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM
from safetensors import safe_open
from huggingface_hub import snapshot_download
from accelerate import init_empty_weights, load_checkpoint_and_dispatch


if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'llama.cpp/gguf-py'))
import gguf

import svd_mistral 

def build_lr_linear(linear):
    
    k = min(linear.in_features, linear.out_features)

    data = linear.weight.data
    odt = data.dtype

    data = data.type(torch.float32)

    U, S, V = torch.linalg.svd(data)

    S = torch.diag(S[:k]) ** 0.5
    U = U[:, :k] @ S
    V = S @ V[:k, :]
    
    return U.type(odt), V.type(odt)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")    
    parser.add_argument("--layer-range-start", type=int, default=0)
    parser.add_argument("--layer-range-end", type=int, default=-1)

    parser.add_argument("--compression-ratio", type=float, required=False)

    parser.add_argument("--find-rank-method", choices=["tol", "E", "Esvd", "bsm", "ht"], required=False)
    parser.add_argument("--tolerance", type=float, required=False, default=0.01)
    parser.add_argument("--threshold", type=float, required=False, default=0.9)

    parser.add_argument("--attn", action='store_true')
    parser.add_argument("--ffn", action='store_true')
    
    parser.add_argument("--query", action='store_true')
    parser.add_argument("--key", action='store_true')
    parser.add_argument("--value", action='store_true')
    parser.add_argument("--up", action='store_true')
    parser.add_argument("--down", action='store_true')
    parser.add_argument("--gate", action='store_true')

    args = parser.parse_args()

    weight_location = snapshot_download(args.repo_id, local_files_only=True)
    config = AutoConfig.from_pretrained(args.repo_id,  
            device_map="auto", 
            offload_folder="offload",  
            offload_state_dict=True,)

    #config.num_hidden_layers = 1
    #config.vocab_size = 5200
    #config.intermediate_size = config.intermediate_size // 256
    #config.hidden_size = 16*(config.hidden_size)// 256


    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    model.tie_weights()
    
    with torch.no_grad():

        model = load_checkpoint_and_dispatch(
            model, weight_location, device_map="auto", offload_folder="offload", offload_state_dict=True
        )        
        
        y1 = model.forward(model.dummy_inputs["input_ids"])

        for i in range(len(model.model.layers)):
            layer = model.model.layers[i]
            mlp = svd_mistral.SVD_MistralMLP(config=model.config)
            attn = svd_mistral.SVD_MistralAttention(config=model.config)

            
            for name, mdl in layer.named_modules():
                print(i, name)
                if name == "self_attn.q_proj":
                    u, v = build_lr_linear(mdl)
                    attn.q_u_proj.weight.data = u
                    attn.q_v_proj.weight.data = v
                    if mdl.bias is not None:
                        attn.q_u_proj.bias.data = mdl.bias.data

                elif name == "self_attn.k_proj":
                    u, v = build_lr_linear(mdl)
                    attn.k_u_proj.weight.data = u
                    attn.k_v_proj.weight.data = v
                    if mdl.bias is not None:
                        attn.k_u_proj.bias.data = mdl.bias.data
                    
                elif name == "self_attn.v_proj":
                    u, v = build_lr_linear(mdl)
                    attn.v_u_proj.weight.data = u
                    attn.v_v_proj.weight.data = v
                    if mdl.bias is not None:
                        attn.v_u_proj.bias.data = mdl.bias.data
                    
                elif name == "self_attn.o_proj":
                    u, v = build_lr_linear(mdl)
                    attn.o_u_proj.weight.data = u
                    attn.o_v_proj.weight.data = v
                    if mdl.bias is not None:
                        attn.o_u_proj.bias.data = mdl.bias.data
                    
                elif name == "mlp.gate_proj":
                    u, v = build_lr_linear(mdl)
                    mlp.gate_u_proj.weight.data = u
                    mlp.gate_v_proj.weight.data = v                
                    if mdl.bias is not None:
                        mlp.gate_u_proj.bias.data = mdl.bias.data
                    
                elif name == "mlp.up_proj":
                    u, v = build_lr_linear(mdl)
                    mlp.up_u_proj.weight.data = u
                    mlp.up_v_proj.weight.data = v
                    if mdl.bias is not None:
                        mlp.up_u_proj.bias.data = mdl.bias.data
                    
                elif name == "mlp.down_proj":
                    u, v = build_lr_linear(mdl)
                    mlp.down_u_proj.weight.data = u
                    mlp.down_v_proj.weight.data = v
                    if mdl.bias is not None:
                        mlp.down_u_proj.bias.data = mdl.bias.data
                    
            model.model.layers[i].self_attn = attn  
            model.model.layers[i].mlp = mlp
                    
        y2 = model.forward(model.dummy_inputs["input_ids"])


        print(((y1["logits"]-y2["logits"])**2).mean())

        torch.save(model, "./model01")
        n1 = torch.load("./model01")

        print(n1)

    exit()


    LAYER_RANGE_START = args.layer_range_start
    if args.layer_range_end == -1:
        LAYER_RANGE_END = config.num_hidden_layers
    else:
        LAYER_RANGE_END = args.layer_range_end
        

    tensor_mapping = gguf.tensor_mapping.get_tensor_name_map(gguf.constants.MODEL_ARCH.QWEN2, config.num_hidden_layers)
    tensorfiles = [fname for fname in os.listdir(weight_location) if fname.endswith(".safetensors")]

    def filter_name(name):

        is_up = lambda s : s == "ffn_up"
        is_down = lambda s : s == "ffn_down"
        is_gate = lambda s : s == "ffn_gate"
        is_query = lambda s : s == "attn_q"
        is_key = lambda s : s == "attn_k"
        is_value = lambda s : s == "attn_v"
        is_attn = lambda s : is_query(s) or is_key(s) or is_value(s)
        is_ffn = lambda s : is_up(s) or is_down(s) or is_gate(s)

        is_weight = lambda s : s.endswith(".weight")

        is_in_range = lambda bid : bid >= LAYER_RANGE_START and bid < LAYER_RANGE_END

        if is_weight(name):
                
            name = name.replace(".weight", "")
            
            _, gguf_tensor_name = tensor_mapping.mapping[name]

            if match := re.match(r"blk\.(\d+)\.(.+)", gguf_tensor_name):
                bid = match.group(1)
                wtype = match.group(2)

                if is_in_range(int(bid)):
                        
                    if args.attn:
                        if is_attn(wtype): return True, gguf_tensor_name

                    if args.ffn:
                        if is_ffn(wtype): return True, gguf_tensor_name
                    
                    if args.query:
                        if is_query(wtype): return True, gguf_tensor_name
                        
                    if args.key:
                        if is_key(wtype): return True, gguf_tensor_name
                        
                    if args.value:
                        if is_value(wtype): return True, gguf_tensor_name

                    if args.up:
                        if is_up(wtype): return True, gguf_tensor_name
                        
                    if args.down:
                        if is_down(wtype): return True, gguf_tensor_name
                        
                    if args.gate:
                        if is_gate(wtype): return True, gguf_tensor_name
                
        return False, ""


    INDEX_FILE = None
    if os.path.exists(indexfile := os.path.join(weight_location,'model.safetensors.index.json')):
        import json
        INDEX_FILE = json.load(open(indexfile))



    exit()
    tensors = {}
    backwards_name_map = {}

    for filename in tensorfiles:
        with safe_open(os.path.join(weight_location,filename), framework="pt") as f:
            for k in f.keys():
                filter_result, gguf_tensor_name = filter_name(k)
                if filter_result:
                    tensor = f.get_tensor(k)
                    tensors[k] = tensor

                    backwards_name_map[gguf_tensor_name] = k
                    print(k, gguf_tensor_name)

                    u, s, v = torch.linalg.svd(tensor.type(torch.float32))

                    vals.append(estimate_rank_TOL(s, 0.05))
                    
                    vals.append(estimate_rank_entropy(s, 0.9))
                    
                    vals.append(estimate_rank_entropy_svd(s, 0.9))
                    
                    #vals.append(estimate_rank_broken_stick(s))
                    
                    vals.append(estimate_rank_hard_threshold(s, tensor.shape[0], tensor.shape[1]))