import asyncio
from datasets import load_dataset
import json
import openai
from tqdm.asyncio import tqdm


system_prompt = '''
我将给你提供一个来源于图片中的名词，你需要根据这个名词，设计一个需要隐式地指向这个名词的语句。

例如：
图片内容是一个储物间，其中有雨伞，锤子，电锯等物品；
我提供给你的名词是：【电锯】
你应该输出：“一个能帮助我切割树木的工具”

注意，这些语句中不得出现该物品本身！并且必须可以经过推断获得答案。

我输入给你的场景和物品是：
'''

input_data = input()
small_batch = [input_data]

async def async_query_openai(query_message):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        base_url="https://api.siliconflow.cn/v1/",
        api_key="sk-xkucgdnxoclxsihlntfibegmmnoscfxpmiuypycbfnttbpvr"
    )

    
    response = await client.chat.completions.create(
        model="deepseek-ai/deepseek-vl2",
        messages=[
            {
                    "role": "user",
                    "content": query_message # 输入给他的东西
            }
        ],
        max_tokens=512,
        temperature=0.01,
        # 下面的都是默认参数没动过
        top_p=0.7,
        # top_k=50,
        frequency_penalty=1,
        # stop=["<|eot_id|>"],
        stream=False
    
    )

    if not response:
        return 'null'
    return response.choices[0].message.content



# 处理单个文本的函数
async def process_text(text):
    prompt = system_prompt + text
    return await async_query_openai(prompt)

# 将结果写入 .jsonl 文件的函数
def write_results_to_jsonl(rawtexts, results, filename):
    with open(filename, 'a', encoding='utf-8') as f:  # 注意改为'a'模式，追加到文件中
        for raw, result in zip(rawtexts, results):
            entry = {
                "origin":raw,
                "response": result
            }
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

# 主函数，运行 asyncio 事件循环和写入结果
async def main():
    import nest_asyncio
    nest_asyncio.apply()
    batch_size = 100
    for i in tqdm(range(0, len(small_batch), batch_size)):
        current_batch = small_batch[i:i+batch_size]
        print(f"Processing batch {i // batch_size + 1}...")
        results = await process_all_texts(current_batch)
        write_results_to_jsonl(current_batch, results, 'results.jsonl')
        print(f"Batch {i // batch_size + 1} written to results.jsonl")
        

# 处理所有文本的函数，调整为处理一批文本
async def process_all_texts(onebatch):
    semaphore = asyncio.Semaphore(10000)
    async def process_one_text(text):
        async with semaphore:
            return await process_text(text)
    
    tasks = []
    for idx in range(len(onebatch)):
        text = onebatch[idx]
        tasks.append(process_one_text(text))
    results = await asyncio.gather(*tasks)  # 使用 gather 等待所有任务完成并保持顺序
    return results


loop = asyncio.get_event_loop()
loop.run_until_complete(main())


import jsonlines
 
file_jsonl_path = "results.jsonl"

extract_results = []

with open(file_jsonl_path, encoding='utf-8') as file:
    for onetest in jsonlines.Reader(file):
        #extract_results.append(onetest['response'])
        print(onetest['response'])
    