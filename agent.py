import os
import logging
import arxiv
import pprint

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.agents.doc_qa import ParallelDocQA
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@register_tool('my_image_gen')
class MyImageGen(BaseTool):
    description = 'AI 绘画（图像生成）服务，输入文本描述，返回基于文本信息绘制的图像 URL。'
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': '期望的图像内容的详细描述',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        import json5
        import urllib.parse
        prompt = json5.loads(params)['prompt']
        prompt = urllib.parse.quote(prompt)
        return json5.dumps(
            {'image_url': f'https://image.pollinations.ai/prompt/{prompt}'},
            ensure_ascii=False)


def download_arxiv_pdfs(query, max_results, directory='pdfs'):
    if not os.path.exists(directory):
        os.makedirs(directory)

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    for result in client.results(search):
        try:
            filename = f"{directory}/{result.get_short_id()}.pdf"
            result.download_pdf(filename=filename)
            logging.info(f"Downloaded: {filename}")
        except Exception as e:
            logging.error(f"Error downloading {result.get_short_id()}: {str(e)}")


def setup_qwen_agent(pdf_directory='pdfs'):
    llm_cfg = {
        'model': 'qwen-long',
        'model_server': 'dashscope',
        'api_key': '',
        'generate_cfg': {
            'top_p': 0.7
        }
    }

    system_instruction = '''你是一个乐于助人的AI助手。
    在收到用户的请求后，你应该：
    - 首先绘制一幅图像，得到图像的url，
    - 然后运行代码`request.get`以下载该图像的url，
    - 最后从给定的文档中选择一个图像操作进行图像处理。
    用 `plt.show()` 展示图像。
    你总是用中文回复用户。'''

    tools = ['my_image_gen', 'code_interpreter']
    files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

    return Assistant(llm=llm_cfg,
                     system_message=system_instruction,
                     function_list=tools,
                     files=files)


def main():
    pdf_directory = 'pdfs'
    # 要查询的论文的内容
    query = "jailbreak"
    max_results = 200

    if not os.path.exists(pdf_directory) or not any(f.endswith('.pdf') for f in os.listdir(pdf_directory)):
        logging.info("No PDFs found. Downloading...")
        download_arxiv_pdfs(query, max_results, pdf_directory)
    else:
        logging.info("PDFs already exist. Skipping download.")

    # Ensure PDFs were downloaded before setting up the agent
    if not any(f.endswith('.pdf') for f in os.listdir(pdf_directory)):
        logging.error("No PDFs found after download attempt. Exiting.")
        return

    bot = setup_qwen_agent(pdf_directory)

    messages = []
    while True:
        try:
            query = input('用户请求: ')
            if query.lower() in ['退出', 'exit', 'quit']:
                break

            messages.append({'role': 'user', 'content': query})
            response = []
            for response in bot.run(messages=messages):
                print('机器人回应:')
                pprint.pprint(response, indent=2)
            messages.extend(response)
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            print("抱歉，处理您的请求时出现了错误。请再试一次。")


if __name__ == "__main__":
    main()
