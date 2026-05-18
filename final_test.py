print("开始导入依赖...")
from transformers.models.autocompressor.modeling_autocompressor import AutoCompressorForCausalLM
from transformers import AutoTokenizer
import torch

print("开始加载模型 (这个过程可能需要 1-2 分钟，请耐心等待)...")
model_path = "./autocompressor_model" 

print("正在加载本地模型...")
model = AutoCompressorForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16,
    local_files_only=True  
).eval().cuda()
print("模型加载完成！")

print("开始加载分词器 (Tokenizer)...")
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/AutoCompressor-Llama-2-7b-6k")
prompt = 'The first name of the current US president is "'
prompt_tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda()

context = """Joe Biden, born in Scranton, Pennsylvania, on November 20, 1942, had a modest upbringing in a middle-class family. He attended the University of Delaware, where he double-majored in history and political science, graduating in 1965. Afterward, he earned his law degree from Syracuse University College of Law in 1968.\nBiden's early political career began in 1970 when he was elected to the New Castle County Council in Delaware. In 1972, tragedy struck when his wife Neilia and 1-year-old daughter Naomi were killed in a car accident, and his two sons, Beau and Hunter, were injured. Despite this devastating loss, Biden chose to honor his commitment and was sworn in as a senator by his sons' hospital bedsides.\nHe went on to serve as the United States Senator from Delaware for six terms, from 1973 to 2009. During his time in the Senate, Biden was involved in various committees and was particularly known for his expertise in foreign affairs, serving as the chairman of the Senate Foreign Relations Committee on multiple occasions.\nIn 2008, Joe Biden was selected as the running mate for Barack Obama, who went on to win the presidential election. As Vice President, Biden played an integral role in the Obama administration, helping to shape policies and handling issues such as economic recovery, foreign relations, and the implementation of the Affordable Care Act (ACA), commonly known as Obamacare.\nAfter completing two terms as Vice President, Joe Biden decided to run for the presidency in 2020. He secured the Democratic nomination and faced the incumbent President Donald Trump in the general election. Biden campaigned on a platform of unity, promising to heal the divisions in the country and tackle pressing issues, including the COVID-19 pandemic, climate change, racial justice, and economic inequality.\nIn the November 2020 election, Biden emerged victorious, and on January 20, 2021, he was inaugurated as the 46th President of the United States. At the age of 78, Biden became the oldest person to assume the presidency in American history.\nAs President, Joe Biden has worked to implement his agenda, focusing on various initiatives, such as infrastructure investment, climate action, immigration reform, and expanding access to healthcare. He has emphasized the importance of diplomacy in international relations and has sought to rebuild alliances with global partners.\nThroughout his long career in public service, Joe Biden has been recognized for his commitment to bipartisanship, empathy, and his dedication to working-class issues. He continues to navigate the challenges facing the nation, striving to bring the country together and create positive change for all Americans."""
context_tokens = tokenizer(context, add_special_tokens=False, return_tensors="pt").input_ids.cuda()

summary_vectors = model(context_tokens, output_soft_prompt=True).soft_prompt
print(f"Compressing {context_tokens.size(1)} tokens to {summary_vectors.size(1)} summary vectors")
# >>> Compressing 660 tokens to 50 summary vectors

# ====== 添加调试信息 ======
print("\n【调试信息】")
print(f"Summary vectors shape: {summary_vectors.shape}")
print(f"Summary vectors 前5个值: {summary_vectors[0, 0, :5]}")
print(f"embed_summary 权重前5个值: {model.embed_summary.weight[0, :5]}")
print(f"Prompt tokens shape: {prompt_tokens.shape}")
print(f"Prompt tokens 内容: {prompt_tokens}")
# ==========================

generation_with_summary_vecs = model.generate(prompt_tokens, do_sample=False, soft_prompt=summary_vectors, max_new_tokens=12)[0]
print("Generation w/ summary vectors:\n" + tokenizer.decode(generation_with_summary_vecs))
# >>> The first name of the current US president is "Joe" and the last name is "Biden".

next_tokens_without_context = model.generate(prompt_tokens, do_sample=False, max_new_tokens=11)[0]
print("Generation w/o context:\n" + tokenizer.decode(next_tokens_without_context))
# >>> The first name of the current US president is "Donald" and the last name is "Trump".