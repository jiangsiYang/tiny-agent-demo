import json5

from InternLM2Chat import InternLM2Chat
from tools import Tools

# demo
"""google_search: Call this tool to interact with the 谷歌搜索 API. What is the 谷歌搜索 API useful for? 谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。 Parameters: [ [
                    {
                        'name': 'search_query',
                        'description': '搜索关键词或短语',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ]] Format the arguments as a JSON object.
"""
TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""
REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""


class Agent:
    def __init__(self, path: str = '') -> None:
        self.path = path
        # 将整个工具类作为一个属性
        self.tool = Tools()
        self.system_prompt = self.build_system_input()
        self.model = InternLM2Chat(path)

    def build_system_input(self):
        tool_descs, tool_names = [], []
        for tool in self.tool.toolConfig:
            tool_names.append(tool['name_for_model'])
            # 将每个工具用TOOL_DESC的模板形式添加到集合中
            tool_descs.append(TOOL_DESC.format(**tool))
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool_names)
        system_prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names)
        return system_prompt

    def parse_latest_plugin_call(self, text):
        plugin_name, plugin_args = '', ''
        # rfing找到特定子字符串最后出现位置
        i = text.rfind('\nAction:')
        j = text.rfind('\nAction Input:')
        k = text.rfind('\nObservation:')
        # 确保输出了且顺序正确
        if 0 <= i < j:
            # 若没有输出Observation，则在文本末尾手动添加
            if k < j:
                text = text.rstrip() + '\nObservation:'
            k = text.rfind('\nObservation')
            # 截取Action和Action Input之间的字符，就是Action的内容了
            plugin_name = text[i + len('\nAction:'): j].strip()
            plugin_args = text[j + len('\nAction Input:'): k].strip()
            # 截断为直到最后的 "\nObservation:" 位置的文本。后面调用工具，会补上调用工具的结果在Observation: 后面
            text = text[:k]
        return plugin_name, plugin_args, text

    def call_plugin(self, plugin_name, plugin_args):
        plugin_args = json5.loads(plugin_args)
        if plugin_name == 'google_search':
            # **plugin_args的方式可以动态地传递参数给到函数，不需要提前显示转换并指定每个参数的名字和值
            return '\Observation:' + self.tool.google_search(**plugin_args)

    def text_completion(self, text, history=[]):
        text = "\nQuestion:" + text
        # 第一轮交互，让LLM知道自己可以调用哪些工具，并以什么样的格式输出
        response, history = self.model.chat(text, history, self.system_prompt)
        plugin_name, plugin_args, response = self.parse_latest_plugin_call(response)
        # 若plugin_name存在，则说明要调用tool
        if plugin_name:
            response += self.call_plugin(plugin_name, plugin_args)
            response, history = self.model.chat(response, history, self.system_prompt)
        # 如果不需要调用工具了，直接返回当前的结果
        return response, history


if __name__ == '__main__':
    agent = Agent(f'E:\huggingface\hub\internlminternlm2-chat-20b')
    prompt = agent.build_system_input()
    print(prompt)
