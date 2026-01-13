import sys, os, json, time, random 
import numpy as np
import argparse
import asyncio
import aiohttp
from tqdm import tqdm

p1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(p1)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from searchagent.agent.agent import AgentInterface

def parse_args():
    parser = argparse.ArgumentParser(description="Run ManuSearch for various datasets and models.")
    parser.add_argument('--single_question', type=str, default=None, help="Single question to process instead of dataset")
    parser.add_argument('--dataset_name', type=str, required=False, default='custom', help="Name of the dataset to use.")
    parser.add_argument('--split', type=str, required=False, default='test', help="Dataset split to use.")
    parser.add_argument('--subset_num', type=int, default=-1, help="Number of examples to process. Defaults to all if not specified.")

    parser.add_argument('--google_subscription_key', type=str, required=True, help="Google Search API subscription key(for serper.dev).")
    parser.add_argument('--google_search_topk', type=int, default=5, help="topk returned documents for google search")
    parser.add_argument('--proxy', type=str, help="port-based proxy(e.g., localhost:8080)")
    parser.add_argument('--planner_model_name', type=str, required=True, help="Name of the planner model to use")
    parser.add_argument('--planner_api_base', type=str, required=True, help="Base URL for the API endpoint")
    parser.add_argument('--planner_api_key', type=str, required=True, help="api key for the planner model API endpoint")
    parser.add_argument('--searcher_model_name', type=str, required=True, help="Name of the searcher model to use")
    parser.add_argument('--searcher_api_base', type=str, required=True, help="Base URL for the API endpoint")
    parser.add_argument('--searcher_api_key', type=str, required=True, help="api key for the searcher model API endpoint")
    parser.add_argument('--reader_model_name', type=str, required=True, help="Name of the reader model to use")
    parser.add_argument('--reader_api_base', type=str, required=True, help="Base URL for the API endpoint")
    parser.add_argument('--reader_api_key', type=str, required=True, help="api key for the reader model API endpoint")
    parser.add_argument('--cache_dir', type=str, required=False, help="cache for searched webpages")
    parser.add_argument('--concurrent_limit', type=int, default=32, help="Maximum number of concurrent API calls")

    parser.add_argument('--temperature', type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument('--top_p', type=float, default=0.95, help="Top-p sampling parameter.")
    parser.add_argument('--min_p', type=float, default=0.0, help="Minimum p sampling parameter.")
    parser.add_argument('--top_k', type=int, default=30, help="Top-k sampling parameter.")
    parser.add_argument('--repetition_penalty', type=float, default=1.05, help="Repetition penalty. If not set, defaults based on the model.")
    parser.add_argument('--max_new_tokens', type=int, default=8192, help="Maximum number of new tokens to generate. If not set, defaults based on the model and dataset.")
    parser.add_argument('--searcher_same_parameters', type=int, default=True, help="Whether adopt the same parameter as planner for searcher.")
    parser.add_argument('--reader_same_parameters', type=int, default=True, help="Whether adopt the same parameter as planner for reader.")

    parser.add_argument('--seed', type=int, default=None, help="Random seed for generation. If not set, will use current timestamp as seed.")
    return parser.parse_args()

args = parse_args()


async def process_single_sequence(agent, message):
    seq = {}

    # Use run_in_executor to execute synchronized methods
    loop = asyncio.get_event_loop()
    steps = await loop.run_in_executor(
        None,  # Default thread pool
        lambda: list(agent.get_answer(message, solve_method='iterative'))  # Convert to list to avoid generator problems
    )

    for step, use_en in steps:
        answer = step.get('final_resp', '')

    think = await loop.run_in_executor(
        None,
        agent.recorder.generate_reason_process
    )

    seq['output'] = answer
    seq['think'] = think
    return seq


async def main_async():
    # Set random seed
    if args.seed is None:
        args.seed = int(time.time())
    random.seed(args.seed)
    np.random.seed(args.seed)  

    # Set Search api
    search_api_keys = [key.strip() for key in args.google_subscription_key.split(",")]
    args.google_subscription_key = search_api_keys

    # Set ManuSearch agent 
    agent = AgentInterface(
        google_subscription_key=args.google_subscription_key,
        google_search_topk=args.google_search_topk,
        proxy=args.proxy,
        planner_model_name=args.planner_model_name,
        planner_api_base=args.planner_api_base,
        planner_api_key=args.planner_api_key,
        searcher_model_name=args.searcher_model_name,
        searcher_api_base=args.searcher_api_base,
        searcher_api_key=args.searcher_api_key,
        reader_model_name=args.reader_model_name,
        reader_api_base=args.reader_api_base,
        reader_api_key=args.reader_api_key,
        my_cache_dir=args.cache_dir,
        temperature=args.temperature,
        top_p=args.top_p, 
        min_p=args.min_p, 
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty, 
        max_new_tokens=args.max_new_tokens,
        searcher_same_parameters=args.searcher_same_parameters,
        reader_same_parameters=args.reader_same_parameters
    )

    # Modified data loading section
    if args.single_question:
        # Create a single item in the same format as dataset items
        filtered_data = [{
            'Question': args.single_question,
        }]
        args.dataset_name = 'custom'  # Set dataset name to custom for single questions
    
    else:
        # Original dataset loading logic
        if args.dataset_name == 'GAIA':
            data_path = f'../data/GAIA/{args.split}.json'
        elif args.dataset_name == 'FRAMES':
            data_path = f'../data/FRAMES/{args.split}.json'
        elif args.dataset_name == 'ORION':
            data_path = f'../data/ORION/{args.split}.json'
        else:
            data_path = f'../data/{args.dataset_name}.json'
        
        print('-----------------------')
        print(f'Using {args.dataset_name} {args.split} set.')
        print('-----------------------')


    # Define output directory
    if 'qwq' in args.planner_model_name.lower():
        model_short_name = 'qwq'
        if 'llama-8b' in args.searcher_model_name.lower():
            model_short_name = 'qwq-llama-8b'
        elif 'llama-70b' in args.searcher_model_name.lower():
            model_short_name = 'qwq-llama-70b'
        elif 'qwen-1.5b' in args.searcher_model_name.lower():
            model_short_name = 'qwq-qwen-1.5b'
        elif 'qwen-7b' in args.searcher_model_name.lower():
            model_short_name = 'qwq-qwen-7b'
        elif 'qwen-14b' in args.searcher_model_name.lower():
            model_short_name = 'qwq-qwen-14b'
        elif 'qwen-32b' in args.searcher_model_name.lower():
            model_short_name = 'qwq-qwen-32b'

    elif 'deepseek' in args.planner_model_name.lower():
        model_short_name = 'dpsk'
        if 'llama-8b' in args.searcher_model_name.lower():
            model_short_name = 'dpsk-llama-8b'
        elif 'llama-70b' in args.searcher_model_name.lower():
            model_short_name = 'dpsk-llama-70b'
        elif 'qwen-1.5b' in args.searcher_model_name.lower():
            model_short_name = 'dpsk-qwen-1.5b'
        elif 'qwen-7b' in args.searcher_model_name.lower():
            model_short_name = 'dpsk-qwen-7b'
        elif 'qwen-14b' in args.searcher_model_name.lower():
            model_short_name = 'dpsk-qwen-14b'
        elif 'qwen-32b' in args.searcher_model_name.lower():
            model_short_name = 'dpsk-qwen-32b'

    else:
        model_short_name = args.searcher_model_name.split('/')[-1].lower().replace('-instruct', '')

    output_dir = f'../outputs/{args.dataset_name}.{model_short_name}.manusearch'
    os.makedirs(output_dir, exist_ok=True)

    
    if not args.single_question:
        # Load and prepare data
        with open(data_path, 'r', encoding='utf-8') as json_file:
            filtered_data = json.load(json_file)

        if args.subset_num != -1:
            indices = list(range(len(filtered_data)))
            selected_indices = random.sample(indices, min(args.subset_num, len(indices)))
            filtered_data = [filtered_data[i] for i in selected_indices]


    # Initialize batch output records
    batch_output_records = []
    start_time = time.time()

    # Create semaphore for concurrent API calls
    semaphore = asyncio.Semaphore(args.concurrent_limit)

    try:
        # Process all sequences concurrently
        tasks = [
            process_single_sequence(
                agent=agent, message=question['Question'],
            )
            for question in filtered_data
        ]

        # Run all sequences concurrently with progress bar
        with tqdm(total=len(tasks)) as pbar:
            async def track_progress(task):
                result = await task
                pbar.update(1)
                return result
            
            tracked_tasks = [track_progress(task) for task in tasks]
            completed_sequences = await asyncio.gather(*tracked_tasks)
    finally:
        pass

    total_time = time.time() - start_time

    t = time.localtime()
    random_num = str(random.randint(0, 99)).zfill(2)
    result_json_name = f'{args.split}-{t.tm_mon}-{t.tm_mday}-{t.tm_hour}-{t.tm_min}-{random_num}.json'
    
    
    # 写入前检查数据
    if not filtered_data:
        print("警告：filtered_data 为空，写入的JSON文件将是空的！")
    else:
        print(f"即将写入 {len(filtered_data)} 条数据到文件")

    for item, seq in zip(filtered_data, completed_sequences):
        item['Output'] = seq['output']
        item['think'] = seq['think']  # Updated field name
    
    
    
    # 写入文件前，添加这些校验代码
    print("===== 数据校验 =====")
    # 1. 打印原始数据（看真实内容）
    print(f"filtered_data 原始内容：{filtered_data}")
    # 2. 检查数据类型
    print(f"filtered_data 类型：{type(filtered_data)}")
    # 3. 尝试手动序列化（模拟json.dump的过程）
    try:
        test_json = json.dumps(filtered_data, ensure_ascii=False)
        print(f"JSON序列化后内容：{test_json}")
        print(f"序列化后字节数：{len(test_json.encode('utf-8'))} 字节")
    except Exception as e:
        print(f"❌ JSON序列化失败：{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    print("====================")



    with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)
        
    # 写入后添加打印
    full_file_path = os.path.join(output_dir, result_json_name)
    print(f"结果已保存到：{full_file_path}")
    # 在原代码的 print("结果已保存到：...") 后添加这两行
    abs_file_path = os.path.abspath(full_file_path)
    print(f"📌 文件绝对路径：{abs_file_path}")  # 打印完整的绝对路径
    print(f"文件是否存在：{os.path.exists(full_file_path)}")
    print(f"文件大小：{os.path.getsize(full_file_path) if os.path.exists(full_file_path) else '不存在'} 字节")


    print("Process completed.")

def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
