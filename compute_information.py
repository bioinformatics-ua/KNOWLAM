from synqgen.ni import HFNIEstimator
import click
import json

@click.command()
@click.argument("collection_file")
@click.argument("model_checkpoint")
@click.option("--context-percentage", type=float, default=0)
@click.option("--context-tokens", type=int, default=0)
def main(collection_file, 
         model_checkpoint, 
         context_percentage,
         context_tokens):

        def read_jsonl(file_path):
            def gen():
                with open(file_path) as f:
                    for data in map(json.loads, f):
                        yield data
                        
            return gen
        
        estimator = HFNIEstimator(model_checkpoint)
        
        _model_name = model_checkpoint.replace("/","_")
        
        _out_name = f"{_model_name}_{collection_file[:-12]}_P{context_percentage}.jsonl"
        
        with open(_out_name, "w") as f:
            for out in estimator.information_from_generator(read_jsonl(collection_file),
                                                            context_percentage=context_percentage,
                                                            context_tokens=context_tokens):
                f.write(f"results/{json.dumps(out)}\n")

if __name__=="__main__":
    main()  