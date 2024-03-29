from synqgen.ni import HFNIEstimator
import click
import json
import torch


MAP = {"float32": torch.float32,
       "float16": torch.float16,
       "bfloat16": torch.bfloat16}

@click.command()
@click.argument("collection_file")
@click.argument("model_checkpoint")
@click.option("--context-percentage", type=float, default=0)
@click.option("--context-tokens", type=int, default=0)
@click.option("--dtype", type=str, default="float32")
def main(collection_file, 
         model_checkpoint, 
         context_percentage,
         context_tokens,
         dtype):

        
        if dtype=="int8":
            _dtype_option = {"load_in_8bit": True}
        elif dtype=="int4":
            _dtype_option = {"load_in_4bit": True}
        else:
            _dtype_option = {"torch_dtype": MAP[dtype]}
        
        def read_jsonl(file_path):
            def gen():
                with open(file_path) as f:
                    for data in map(json.loads, f):
                        yield data
                        
            return gen
        
        estimator = HFNIEstimator(model_checkpoint, 
                                  cache_dir="hf_cache",
                                  model_kwargs={
                                      "device_map": 0,
                                      **_dtype_option
                                  })
        
        _model_name = model_checkpoint.replace("/","_")
        _dataset_name = collection_file[:-12].replace("/","_")
        _out_name = f"{_model_name}_{_dataset_name}_P{context_percentage}_{dtype}.jsonl"
        
        with open(f"results/{_out_name}", "w") as f:
            for out in estimator.information_from_generator(read_jsonl(collection_file),
                                                            context_percentage=context_percentage,
                                                            context_tokens=context_tokens,
                                                            max_samples=10_000,
                                                            max_documents=10_000):
                #print(out)
                f.write(f"{json.dumps(out)}\n")

if __name__=="__main__":
    main()  