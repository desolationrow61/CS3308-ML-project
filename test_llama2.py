import os                          
import torch                       
from datasets import load_dataset  
from transformers import (
    AutoModelForCausalLM,          
    AutoTokenizer,                
    BitsAndBytesConfig,           
    HfArgumentParser,             
    TrainingArguments,            
    pipeline,                     
    logging,                      
)
from peft import LoraConfig, PeftModel  
from trl import SFTTrainer      
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained(
    "/mnt/sda1/home/kailexiangzi/verilog/results_2/checkpoint-5580/",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
merged_model = model.merge_and_unload()


# model_name="/mnt/sda1/home/kailexiangzi/.cache/huggingface/hub/models--TinyPixel--Llama-2-7B-bf16-sharded/snapshots/8c3544353d97b90748a1198bd194996558a2b4e6"
# tokenizer=AutoTokenizer.from_pretrained("/mnt/sda1/home/kailexiangzi/verilog/results/checkpoint-7700")
tokenizer=AutoTokenizer.from_pretrained("/mnt/sda1/home/kailexiangzi/verilog/results_2/checkpoint-5580/")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

use_4bit = True

# Compute dtype for 4-bit base models
# bnb_4bit_compute_dtype = "float16"

# # Quantization type (fp4 or nf4)
# bnb_4bit_quant_type = "nf4"
# compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
# use_nested_quant = False
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=use_4bit,
#     bnb_4bit_quant_type=bnb_4bit_quant_type,
#     bnb_4bit_compute_dtype=compute_dtype,
#     bnb_4bit_use_double_quant=use_nested_quant,
# )


# merged_model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map='auto'
# )

# peft_model_id="/mnt/sda1/home/kailexiangzi/verilog/results/checkpoint-7800"
# model1=PeftModel.from_pretrained(model,peft_model_id)
# model1= AutoModelForCausalLM.from_pretrained(
#     "/mnt/sda1/home/kailexiangzi/verilog/results/checkpoint-7600",
#     # quantization_config=bnb_config,
#     device_map='auto'
# )
# model1.eval()

# def tokenize_function(example):
#     promp_template='### System Prompt: {system_prompt}### Instruction: {instruction}### Output:' 
            
#     text_prop=promp_template.format(system_prompt=example["system_prompt"],instruction=example["instruction"])
#     return text_prop

# dataset_name = "/mnt/sda1/home/kailexiangzi/.cache/huggingface/datasets/emilgoh___verilog-dataset-v2/default-9ff9904a703b3146/0.0.0/eea64c71ca8b46dd3f537ed218fc9bf495d5707789152eb2764f5c78fa66d59d"
# dataset = load_dataset(dataset_name, split="train")
# print(dataset)
# prompt = dataset["prompt"][0]
# promp_template='### System Prompt: {system_prompt}### Instruction: {instruction}### Output:'
# promp_template='{system_prompt} {instruction}.'
# text_prop=promp_template.format(system_prompt=dataset["system_prompt"][1],instruction=dataset["instruction"][1])
text_prop="""### System Prompt: I want you to act as an IC designer, and implement the following in Verilog.### Instruction: Generate a Verilog module with the following description: Module with ANDNOT operation.### Output:"""
inputs = tokenizer(text_prop, return_tensors="pt")
text_prop2="""### System Prompt: I want you to act as an IC designer, and implement the following in Verilog.### Instruction: Generate a Verilog module with the following description: 3-bit full adder module.### Output:"""
text_prop3="""### System Prompt: I want you to act as an IC designer, and implement the following in Verilog.### Instruction: Generate a Verilog module with the following description: Multiplexer module with three inputs.### Output:"""
inputs2=tokenizer(text_prop2, return_tensors="pt")
inputs3=tokenizer(text_prop3,return_tensors="pt")
device="cuda:0"
inputs = inputs.to(device)
inputs2=inputs2.to(device)
inputs3=inputs3.to(device)
        

with torch.no_grad():
    outputs=merged_model.generate(**inputs,max_new_tokens=512)
    print(tokenizer.decode(outputs[0],skip_special_tokens=True))
    print("over")
    outputs=merged_model.generate(**inputs2,max_new_tokens=512)
    print(tokenizer.decode(outputs[0],skip_special_tokens=True))
    print("over")
    outputs=merged_model.generate(**inputs3,max_new_tokens=512)
    print(tokenizer.decode(outputs[0],skip_special_tokens=True))