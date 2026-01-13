import re, json, logging, traceback, copy
from typing import Dict
from pydantic import Field
from ..models.basellm import BaseStreamingAgent
from ..models.reader import Reader
from ..models.recorder import Recorder
from ..schema import AgentMessage 
from ..utils.utils import *
from ..tools.tool_collection import ToolCollection
from ..tools.final_answer import FinalAnswerTool
from ..tools.visitpage import VisitPage
from ..tools.websearch import GoogleSearch


def extract_reference_numbers(ref):
        # 步骤1：确保 ref 是字符串类型
        if ref is None:
            ref = ""
        elif not isinstance(ref, str):
            # 若为字典，提取文本字段；其他类型转为字符串（兜底）
            if isinstance(ref, dict):
                ref = ref.get('summ', ref.get('content', json.dumps(ref, ensure_ascii=False)))
            else:
                ref = str(ref)
    
        # 步骤2：正则匹配并安全转换为整数
        numbers = []
        try:
            # 匹配 [[数字]] 格式的内容
            matches = re.findall(r"\[\[(\d+)\]\]", ref)
            # 转换为整数并去重（集合）
            numbers = list({int(n) for n in matches})
        except (ValueError, TypeError) as e:
            # 匹配/转换失败时返回空列表
            print(f"提取引用数字失败：{e}")
            numbers = []
    
        return numbers


class Searcher(BaseStreamingAgent):
    """A module responsible for parsing and summarizing relevant information from search results."""
    def __init__(
        self,
        llm,
        reader: Reader,
        collected_tools: ToolCollection = None,
        user_input_template: str = "{question}",
        user_context_template: str = None,
        max_turn: int = 10,
        max_length = 24576,
        **baseconfig
    ):
        """
        A module responsible for parsing and summarizing relevant information from search results.

        Args:
            user_input_template (`str`): A template string for formatting the user input. Defaults to "{question}".
            user_context_template (`str`): A template string for formatting the context of the user input. Defaults to None.
            **baseconfig: Additional configuration parameters passed to the base class.

        Methods:
            read(question, search_result, recorder, session_id=0, **kwargs): Summarizes the search results and extracts references.
            parse(search_result): Placeholder method for parsing search results.
            _update_ref(ref, ref2url, ptr): Updates reference indices in the response.
            _generate_references_from_graph(query, response, reference): Generates references from the search results.
        """ 
        self.reader = reader
        self.user_input_template = user_input_template
        self.user_context_template = user_context_template
        self.ptr=0 
        if 'qwen' in llm.model_type.lower():
            self.max_length = 24576
        else:
            self.max_length = 128000
        if collected_tools:
            self.collected_tools = collected_tools
        else:
            self.collected_tools: ToolCollection = ToolCollection(GoogleSearch(), FinalAnswerTool())
            
        self.tools_schema = [tool.to_schema() for tool in self.collected_tools.tools]
        self.search_results = {}
        self.url_to_chunk_score = {}
        super().__init__(llm=llm, max_turn=max_turn, **baseconfig)

    def search(
        self,
        question, # 当前问题 
        recorder, 
        session_id:int =0, 
        **kwargs
    ):
        """
        Parses and summarizes the search results with extracted references.
        If the search engine summary doesn't contain enough info, the crawler will crawl the most importangt pages 
        and llm will regenerate summary for current question. 

        Args:
            question (str): The sub-question being addressed.
            search_result (str): The raw search results from the Searcher.
            recorder (Recorder): The Recorder instance for maintaining search state.
            session_id (int): The session ID for tracking the search process. Defaults to 0.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing the summarized references and a dictionary of reference URLs.
        """     

        def prepare_search(node_name, recorder):
            """
            Args:
                node_name(`str`): The current subqury
                recorder(`recorder`)
            Return:
                topic(`str`): The main query
                history(`List[dict]`): Tha answer of the parent subqueries
            """            
            # 获取父节点，以获得历史对话信息
            parent_nodes = []
            nodes = recorder.container['content'].nodes
            for pre_node_name in nodes.keys():
                if  pre_node_name == 'root':
                    pass
                elif pre_node_name == node_name:
                    break
                else:
                    parent_nodes.append((pre_node_name, nodes[pre_node_name]))

            parent_response = [
                dict(question=node_name, answer=node['response']) for node_name, node in parent_nodes
            ]

            return nodes['root']['content'], parent_response
        
        topic, history = prepare_search(node_name=question, recorder=recorder)
        message = [self.user_input_template.format(question=question)]
        if history and self.user_context_template:
            message = [self.user_context_template.format_map(item) for item in history] + message
        message = "\n".join(message)

        # searcher每轮求解后清空memory
        self.agent.memory.reset(0)
        whether_exceed_max_tokens = False
        messages = [AgentMessage(sender="user", content=message)]
        try:
            for turn in range(self.max_turn):
                if turn == self.max_turn-1:
                    messages.append({
                        "role": "user",
                        "content": "Maximum number of rounds exceeded, please call final answer tools immediately based on information already collected"                    
                    })
                ignore = False

                with timeit("searcher inference"):
                    references = ""
                    references_url = {}
                    for response in super().forward(messages, tools=self.tools_schema, tool_choice="auto", session_id=session_id):

                        if isinstance(response.content, str) and response.content:
                            yield 'model_response', response.content, {}
 
                        elif response.content:
                            tools_in_resp, url2title, query_list = [], {}, []
                            for tool in response.content.tool_calls:
                                name = get_tool_name(tool)
                                tools_in_resp.append(name)
                                if name == 'final_answer' and turn == 0 and ('visitpage' in tools_in_resp or 'GoogleSearch' in tools_in_resp):
                                    ignore = True   
                                    continue
                                arg = get_tool_arg(tool)
                                resp = parse_resp_to_json(arg)
                                if resp and isinstance(resp, dict):
                                    if name == 'final_answer':
                                        if recorder.container['content'].nodes[question]['memory']:
                                            references, references_url = self._generate_references_from_graph(
                                                response=resp.get('answer', ''),
                                                ref2url=recorder.container['content'].nodes[question]['memory'],
                                            )
                                        else:
                                            references, references_url = resp, {}

                                        recorder.update(
                                            node_name=question,
                                            node_content=None,
                                            content= references,
                                            memory=None,
                                            sender='searcher_response'
                                        )
                                        yield 'final_answer', references, references_url
                                        
                                    elif name == 'GoogleSearch':
                                        query_list.extend(resp.get('query', []))

                

                messages = []
                # Tool calls
                if not isinstance(response.content, str):
                    debugs = [f"arguments: {toolcall.function.arguments}, name: {toolcall.function.name}" for toolcall in response.content.tool_calls]
                    print(debugs)
                    for tool_call in response.content.tool_calls:
                        name = tool_call.function.name
                        args = load_multiple_dict(tool_call.function.arguments)
                        print("searcher_line204,args:", args)
                        query_keys = [k for k in args.keys() if 'query' in k.lower()]  # 找到所有包含 query 的键
                        if query_keys:
                            # 将第一个包含 query 的键的值赋值给标准的 query 键
                            args['query'] = args.pop(query_keys[0])
                        else:
                            # 若没有找到 query 相关键，抛出异常或设默认值
                            args['query'] = []
                            print(f"[展平后的 query] {args['query']}")
                        print("searcher_line222,args:", args)
                        if name:
                            if name.lower() == 'googlesearch':
                                with timeit("search web && get summary"):
                                    if 'intent' in args:
                                       if isinstance(args['intent'], list):
                                            args['intent'] = ' '.join(args['intent'])
                                    else:
                                        args['intent'] = "" 
                                    all_argumens = copy.deepcopy(list(args.keys()))
                                    print("all_argumens:", all_argumens)
                                    for key in all_argumens:
                                        if key not in ['query', 'intent']:
                                            args.pop(key)
                                    print("searcher_line216", args)
                                    search_results = self.collected_tools.execute(name=name, tool_input=args)
                                    yield 'webpages', search_results, {}
                                    summ_provided_by_engine = next(({key: item} for key, item in search_results.items() if item['url'] == ""), None)
                                    if summ_provided_by_engine:
                                        part_result = dict(list({key: item for key, item in search_results.items() if item['url']}.items())[:4])
                                        result_list = list(summ_provided_by_engine.values())+list(part_result.values())
                                        search_results = {i: value for i, value in enumerate(result_list)}
                                        search_results, cur_url_to_chunk_score = self.reader.get_llm_summ(search_results, question, topic, args['intent'], args['query'])
                                    else:
                                        search_results, cur_url_to_chunk_score = self.reader.get_llm_summ(search_results, question, topic, args['intent'], args['query'])

                                if self.search_results:
                                    search_results = {key+len(self.search_results):value for key, value in search_results.items()}

                                self.search_results.update(search_results)
                                if isinstance(cur_url_to_chunk_score, dict):
                                    self.url_to_chunk_score.update(cur_url_to_chunk_score)
                                # web info return to LLM: {url, title, summ, date}
                                web_result = {k: {key: val for key, val in v.items()} for k, v in search_results.items()}
                                result = json.dumps(web_result, ensure_ascii=False)
                                recorder.update(
                                    node_name=question,
                                    node_content=args['query'],
                                    content=web_result,
                                    memory=self.agent.memory,
                                    sender='searcher'
                                )
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": copy.deepcopy(result)
                                })

                            elif name.lower() == 'final_answer' and not ignore:
                                if references:
                                    self.ptr+=len(references_url)
                                    result = resp
                                    recorder.update(
                                        node_name=question,
                                        node_content=None,
                                        content= references,
                                        memory=self.agent.memory,
                                        sender='searcher_response'
                                    )
                                    return references
                                else:
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": "Cannot execute this function call. Please retry!"
                                    }) 
                                    
                            else:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": "Based on the search results, Please answer the question again."
                                })
                                

                        else:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": "Cannot execute this function call. Please retry!"
                            })

                # No tool calls, model response
                else:
                    pass
                
        except Exception:
            print('Stack trace:', traceback.format_exc())
            raise

    
        
    def _update_ref(self, ref: str, ref2url: Dict[str, str], ptr: int) -> str:

        """
        Updates references within a given string based on reference indices and the provided pointer.

        Args:
            ref (`str`): The reference string that needs updating.
            ref2url (`dict`): A dictionary of reference indices and their associated URLs.
            ptr (`int`): The current pointer to update references with.

        Returns:
            tuple:
                - updated_ref (`str`): The updated reference string with modified indices.
                - updated_ref2url (`dict`): A dictionary mapping updated reference indices to their URLs.
                - added_ptr (`int`): The number of new references added.
            
        """            
        print(ref)
        
        if isinstance(ref, dict):
            # 核心修改：提取answer字段，不存在则用空字符串兜底
            ref = ref.get('answer', '')
        elif ref is None:
            # 如果是None，初始化为空字符串
            ref = ""
        elif not isinstance(ref, (str, bytes)):
            # 其他类型，强制转换为字符串
            ref = str(ref)
        #numbers = list({int(n) for n in re.findall(r"\[\[(\d+)\]\]", ref)})
        numbers = extract_reference_numbers(ref)

        if not numbers:
            return ref, {}
        numbers = {n: idx + 1 for idx, n in enumerate(numbers)}
        updated_ref = re.sub(
            r"\[\[(\d+)\]\]",
            lambda match: f"[[{numbers[int(match.group(1))] + ptr}]]",
            ref,
        )
        updated_ref2url = {}
        try:
            assert all(elem in ref2url for elem in numbers)
        except Exception as exc:
            logging.info(f"Illegal reference id: {str(exc)}")
        if ref2url:
            updated_ref2url = {
                numbers[idx] + ptr: ref2url[idx] for idx in numbers if idx in ref2url
            }
        return updated_ref, updated_ref2url


    def _generate_references_from_graph(self, response, ref2url) -> tuple[str, Dict[int, dict]]:
        """
        Generates references from the search result graph and updates the reference indices.

        Args:
            query (`str`): The original query or question.
            response (`str`): The summarized response based on the search result.
            reference (`str`): The reference string containing previous references.

        Returns:
            tuple:
                - references (`str`): The formatted reference string.
                - references_url (`dict`): A dictionary of references with their corresponding URLs.
        """
        if not ref2url:
            return response, {}
        updata_ref, ref2url = self._update_ref(
            response, ref2url, self.ptr
        )
        return updata_ref, ref2url


