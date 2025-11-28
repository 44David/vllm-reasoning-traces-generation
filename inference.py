import json
from vllm import LLM, SamplingParams
from datasets import load_dataset
from tqdm import tqdm 
import sys

def main():

    if len(sys.argv[1]) > 1:
        try:
            ds = load_dataset(sys.argv[1])
        except:
            print("not a valid hf dataset, please enter author/dataset_name")
            return
    else:
        print("must provide a hf dataset: author/dataset_name")
        return
        
        
    if len(sys.argv[2]) > 1:
        samples_to_process = int(sys.argv[2])
    else: 
        print("must have a number of samples to process")
        return
            
    if len(sys.argv[3]) > 1:
        output_file = sys.argv[3]    


    llm = LLM(model="deepseek-ai/deepseek-math-7b-instruct")

    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=1024,
        top_p=0.9
    )

    progress_bar = tqdm(total=samples_to_process)

    for i in range(0, samples_to_process, 1):
        prompt = f"{ds['train']['problem'][i]} \n Please reason step by step, and put your final answer within \\\\boxed{{}}."

        outputs = llm.generate([prompt] * 6, sampling_params)

        solutions = []
        for output in outputs:
            full_response = output.outputs[0].text
            
            # check if output prompt is too short or doesn't contain an actual answer.
            while (len(full_response) < 5 or !full_response.contains("\\\\boxed")):
                out = llm.generate(prompt, sampling_params)
                full_response = out.outputs[0].text
            
            solutions.append(full_response)


        data_point = {
            "problem": ds["train"]["problem"][i],
            "thinking_traces": solutions,
            "correct_answer": ds["train"]["solution"][i],
            "subject": ds["train"]["type"][i],
            "level": ds["train"]["level"][i]
        }

        with open(output_file, "a") as f:
            f.write(json.dumps(data_point) + "\n")

        progress_bar.update(1)
    
main()