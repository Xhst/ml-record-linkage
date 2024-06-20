from llama_cpp import Llama
import paths
llm = Llama(model_path=paths.MODELS_DIR + "/Llama3-70B/L3-70B-Euryale-v2.1-Q5_K_M.gguf", n_gpu_layers=30, n_ctx=3584, n_batch=521, verbose=True)
# adjust n_gpu_layers as per your GPU and model
output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
print(output)