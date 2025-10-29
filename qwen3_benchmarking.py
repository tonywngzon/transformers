import os
import time
import argparse
import json
import shutil
from tqdm import tqdm
import torch
import torch_neuronx
from transformers import AutoTokenizer, AutoModel

#os.environ['NEURON_RT_INSPECT_ENABLE']="0"
os.environ['NEURON_RT_VIRTUAL_CORE_SIZE']="1"
os.environ['NEURON_LOGICAL_NC_CONFIG']="1"
os.environ['NEURON_RT_NUM_CORES']="1"
#os.environ['NEURON_RT_VISIBLE_CORES']="0-1"

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-Embedding-0.6B", padding_side="left"
)
model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
# for name, param in model.state_dict().items():
#     print(f"{name}: {param.shape}")
model.config.use_qkv_kernel = True
model.config.fused_qkv = True
model.config.fused_rmsnorm = True

# Set attributes directly on attention layers
# print("Model structure:", type(model))
# print("Model attributes:", dir(model))
if hasattr(model, 'layers'):
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'self_attn'):
            print(f"Setting attributes on layer {i}")
            layer.self_attn.qkv_kernel_enabled = True
            layer.self_attn.fused_qkv = True
            layer.self_attn.fused_rmsnorm = True

model.eval()


# Create a wrapper to return the last hidden state
class NeuronQwenEmbedding(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask)
        return output.last_hidden_state

# Function to generate batched and padded inputs
def encode(tokenizer, *inputs, max_length=32, batch_size=1):
    tokens = tokenizer.encode_plus(
        *inputs,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    return (
        torch.repeat_interleave(tokens['input_ids'], batch_size, 0),
        torch.repeat_interleave(tokens['attention_mask'], batch_size, 0),
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--visible_cores', type=str, help='Range of visible cores')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--max_length', type=int, help='Max length')
    parser.add_argument('--saved_dir', type=str, default='saved_models', help='Directory to save/load compiled models')
    parser.add_argument('--force_recompile', default='false', action='store_true', help='Force recompilation even if saved model exists')

    args = parser.parse_args()

    os.environ['NEURON_RT_VISIBLE_CORES']=args.visible_cores

    # Set up inputs
    # TODO - Add input replication for BS > 1
    batch_size = args.batch_size
    max_length = args.max_length
    saved_dir = args.saved_dir
    force_recompile = args.force_recompile

    print(f"{batch_size=}, {max_length=}")


    text = "goldfish"*100000
    
    example = encode(tokenizer, text, max_length=max_length, batch_size=batch_size)
    # print(example)
    # print(sum(list(example[1].numpy()[0]))) # making sure nothing is masked cuz not sure if i trust the attn mask

    # exit(example[0].shape)

    # # Check CPU outputx
    # golden = model(*example)
    # print("CPU:")
    # print(golden)

    # # Create wrapped model output
    model_wrapped = NeuronQwenEmbedding(model)
    model_wrapped.eval()
    # # Validate output matches
    # model_wrapped_output = model_wrapped(*example)
    # print("CPU Model wrapped: ")
    # print(model_wrapped_output)

    # # Confirm the wrapper produces the same output
    # torch.testing.assert_allclose(
    #     golden.last_hidden_state, model_wrapped_output, rtol=0, atol=0
    # )

    # Setup model file path
    filename = f"Qwen3-Embedding-Neuron-seqlen{max_length}-bs{batch_size}.pt"
    model_file = os.path.join(saved_dir, filename)
    compiler_workdir = f"workdir_Qwen3-Embedding_s{max_length}_b{batch_size}"
    
    # Check if saved model exists and load it, or compile new model
    if os.path.isfile(model_file) and not force_recompile:
        print(f"Load model from {model_file}")
        model_neuron = torch.jit.load(model_file)
        # When inline_weights_to_neff=False we need manually load the weights onto device
        torch_neuronx.move_trace_to_device(model_neuron, 0)
    else:
        print(f"Trace model and save it to {model_file}")
        os.makedirs(saved_dir, exist_ok=True)
        shutil.rmtree(compiler_workdir, ignore_errors=True)
        shutil.rmtree(model_file, ignore_errors=True)
        # Compile
        print("Compiling the model for Neuron ...")
        # Use -O1 to avoid too many instructions error
        # Use inline_weights_to_neff=False to avoid baking the weights in to the NEFF
        model_neuron = torch_neuronx.trace(
            model_wrapped,
            example,
            compiler_args="-O1 --logical-nc-config=1 --internal-hlo2tensorizer-options=' --partitioner-split-opcodes=dot'",
            inline_weights_to_neff=False,
            compiler_workdir=compiler_workdir,
        )
        print("Compilation successful! Saving the model ...")
        
        # Save
        torch.jit.save(model_neuron, model_file)
        
        # Load
        model_neuron = torch.jit.load(model_file)
        # When inline_weights_to_neff=False we need manually load the weights onto device
        torch_neuronx.move_trace_to_device(model_neuron, 0)

    # Check Neuron output
    actual = model_neuron(*example)
    print("Neuron:")
    print(actual)

    # Benchmark
    iters = 100

    start = time.time()
    for _ in tqdm(range(iters)):
        # run inference
        output = model_neuron(*example)

    total_time = time.time() - start
    throughput = (batch_size * iters) / total_time

    benchmark_data = {
        "Batch Size": batch_size,
        "Throughput": throughput,
        "Instance Throughput": 128 * throughput
    }
    with open(f"benchmarks/{batch_size}.json", "w") as f:
        json.dump(benchmark_data, f)

    print(f"Throughput: {throughput} prompts/s")
    print(f"Instance throughput: {128 * throughput} prompts/s")
