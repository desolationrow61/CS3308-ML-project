from langchain import HuggingFacePipeline,PromptTemplate,  LLMChain
import os                          
import torch                       
from datasets import load_dataset,load_from_disk
from transformers import (
    AutoModelForCausalLM,          
    AutoTokenizer,                
    BitsAndBytesConfig,           
    HfArgumentParser,             
    TrainingArguments,            
    pipeline,                     
    logging,               
    pipeline       
)
from peft import LoraConfig, PeftModel  
from trl import SFTTrainer      
from peft import AutoPeftModelForCausalLM
from langchain.memory import ConversationBufferMemory

# dataset_name = "/mnt/sda1/home/kailexiangzi/.cache/huggingface/datasets/emilgoh___verilog-dataset-v2/default-9ff9904a703b3146/0.0.0/eea64c71ca8b46dd3f537ed218fc9bf495d5707789152eb2764f5c78fa66d59d"
# dataset = load_dataset(dataset_name, split="train")
# print(dataset.num_rows)
# dataset = dataset.filter(lambda x: len(x["prompt"]) < 512)
# print(dataset.num_rows)
# dataset=dataset.train_test_split(train_size=0.9,seed=42)
# print(dataset)
# dataset.save_to_disk("/mnt/sda1/home/kailexiangzi/verilog/dataset/")
# dataset.load_from_disk()

# dataset=load_from_disk("/mnt/sda1/home/kailexiangzi/verilog/dataset/")
# print(dataset)
model = "/mnt/sda1/home/kailexiangzi/.cache/huggingface/hub/models--TinyPixel--Llama-2-7B-bf16-sharded/snapshots/8c3544353d97b90748a1198bd194996558a2b4e6"
# print(dataset["test"]["prompt"][177])
# print(dataset["test"]["prompt"][1005])
# print(dataset["test"]["prompt"][188])
tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
pipeline = pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_new_tokens=512,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})
template = """Generate verilog codes for given input.
-----------
[Q]: Generate a Verilog module with the following description: 3-input NOR gate module declaration.
[A]: module JNOR3A(A1, A2, A3, O);
input   A1;
input   A2;
input   A3;
output  O;
nor g0(O, A1, A2, A3);
endmodule
Use above structure to generate code for {text}
"""



prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)
text = """[Q]: Generate a Verilog module with the following description: Module with ANDNOT operation."""
text2 = """[Q]: Generate a Verilog module with the following description: 3-bit full adder module."""
text3 = """[Q]: Generate a Verilog module with the following description: Multiplexer module with three inputs."""
print(template.format(text=text))
ans = llm_chain.run(text=text)

print(ans.replace("\n\n",""))
print("over")

ans = llm_chain.run(text=text2)

print(ans.replace("\n\n",""))
print("over")

ans = llm_chain.run(text=text3)

print(ans.replace("\n\n",""))
print("over")

# template = """Generate verilog codes for given input.
# -----------
# [Q]: Generate a Verilog module with the following description: Simple 6-bit MUX module definition.
# [A]: module module_a (out0,in0);

# input           in0;
# output [5:0]    out0;

# parameter [5:0] ident0 = 0;
# parameter [5:0] ident1 = 5'h11;

# reg [5:0] out0;

# // Basic MUX switches on in0
# always @ (in0)
#    begin
#      if(in0)
#        out0 = ident0;
#      else
#        out0 = ident1;
#    end

# endmodule
# """

# # prompt = PromptTemplate(template=template, input_variables=["text"])

# # llm_chain = LLMChain(prompt=prompt, llm=llm)

# model = AutoModelForCausalLM.from_pretrained(
#     model,
#     low_cpu_mem_usage=True,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )
# que_pre=""
# ans_pre=""
# last= """Use above structure to generate code for {text}"""
# test1="Generate a Verilog module with the following description: Module with ANDNOT operation."
# test2="Generate a Verilog module with the following description: 3-bit full adder module."
# test3="Generate a Verilog module with the following description: Multiplexer module with three inputs."
# for i in range(3):

#     template+=que_pre+ans_pre
#     if i==0:
#         template1=template+last.format(text="[Q]: "+test1) 
#         add_q=test1
#     if i==1:
#         template1=template+last.format(text="[Q]: "+test2)
#         add_q=test2
#     if i==2:
#         template1=template+last.format(text="[Q]: "+test3)
#         add_q=test3
#     print("input:")
#     print(template1)
#     with torch.no_grad():
#         inputs = tokenizer(template1, return_tensors="pt")
#         inputs = inputs.to("cuda")
#         outputs=model.generate(**inputs,max_new_tokens=512)
#         ans=tokenizer.decode(outputs[0],skip_special_tokens=True)
#         ans=ans.strip('\n')
#     ans = ans[len(template1):]
#     ind = ans.index("endmodule")
#     ans=ans[:ind+len("endmodule")]+"\n"
#     print("ans:")
#     print(ans)
#     que_pre="""-----------
# [Q]: {q}""".format(q=add_q)
#     ans_pre=ans
    


