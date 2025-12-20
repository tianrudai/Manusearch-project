from serpapi import GoogleSearch
from typing import Optional, Dict, Any

class SerpAPIClient:
    """SerpAPI客户端类，封装Google搜索调用"""
    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, num_results: int = 10, lang: str = "en-US", location: str = "us") -> Optional[Dict[str, Any]]:
        """
        调用SerpAPI执行Google搜索
        :param query: 搜索关键词
        :param num_results: 返回结果数（最大100）
        :param lang: 搜索语言（如en-US、zh-CN）
        :param location: 地理区域（如us、cn、uk）
        :return: 搜索结果字典，失败返回None
        """
        # 配置搜索参数
        params = {
            "q": query,  # 搜索关键词
            "api_key": self.api_key,  # SerpAPI密钥
            "num": num_results,  # 返回结果数
            "hl": lang,  # 界面语言
            "gl": location,  # 地理区域
            "engine": "google",  # 搜索引擎（固定为google）
            "google_domain": "google.com"  # Google域名（可改为google.com.hk、google.co.uk等）
        }

        try:
            # 执行搜索
            search = GoogleSearch(params)
            results = search.get_dict()  # 获取字典格式的结果（也可用get_json()获取JSON字符串）
            return results

        except Exception as e:
            print(f"SerpAPI调用失败：{str(e)}")
            return None

    def parse_results(self, result: Dict[str, Any]) -> None:
        """解析并格式化打印搜索结果"""
        if not result:
            print("无搜索结果")
            return

        # 提取搜索元信息
        search_info = result.get("search_information", {})
        total_results = search_info.get("total_results", 0)
        search_time = search_info.get("time_taken_displayed", 0)

        print(f"🔍 搜索关键词：{result.get('search_parameters', {}).get('q', '未知')}")
        print(f"📊 找到约 {total_results} 条结果（耗时 {search_time} 秒）\n")

        # 遍历自然搜索结果（排除广告、图片等）
        for idx, item in enumerate(result.get("organic_results", []), 1):
            title = item.get("title", "无标题")
            link = item.get("link", "无链接")
            snippet = item.get("snippet", "无摘要")
            print(f"【{idx}】{title}")
            print(f"链接：{link}")
            print(f"摘要：{snippet}\n")

# ------------------- 调用示例 -------------------
if __name__ == "__main__":
    # 替换为你的SerpAPI密钥
    SERP_API_KEY = "16f849c9136cdc974e6032d9b58b16b74d3ed2f0"

    # 初始化客户端
    client = SerpAPIClient(api_key=SERP_API_KEY)

    # 执行搜索（示例：搜索"人工智能 最新进展"，中文结果）
    search_result = client.search(
        query="今天的日期",
        num_results=5,
        lang="zh-CN",
        location="cn"
    )

    # 解析结果
    client.parse_results(search_result)