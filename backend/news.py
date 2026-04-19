"""
新闻→公寓事件翻译器。
抓取今日新闻标题，用 LLM 映射成哪吒家里能发生的具体事件。
"""
from __future__ import annotations
import json
import re
import xml.etree.ElementTree as ET

import httpx
import anthropic

_MODEL = "claude-sonnet-4-6"

# 免费 RSS 源（无需 API Key）
NEWS_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
]


async def fetch_headlines(max_items: int = 12) -> list[str]:
    """从 RSS 源抓取新闻标题，失败的源跳过。"""
    headlines: list[str] = []
    async with httpx.AsyncClient(timeout=8, follow_redirects=True) as client:
        for url in NEWS_FEEDS:
            if len(headlines) >= max_items:
                break
            try:
                resp = await client.get(url)
                root = ET.fromstring(resp.text)
                for item in root.findall(".//item"):
                    title = item.find("title")
                    if title is not None and title.text:
                        headlines.append(title.text.strip())
                    if len(headlines) >= max_items:
                        break
            except Exception:
                continue
    return headlines[:max_items]


async def translate_to_apartment_events(headlines: list[str]) -> list[dict]:
    """
    把新闻标题映射为哪吒公寓里能发生的具体事件（1-3 个）。
    返回格式与 NARRATIVE_EVENTS 中的 dict 兼容，附加 fire_at="immediate"。
    """
    if not headlines:
        return []

    client = anthropic.AsyncAnthropic()
    prompt = f"""你是《罗小黑战记》除夕夜模拟剧情的编剧助理。

今天的真实新闻标题：
{chr(10).join(f"- {h}" for h in headlines)}

现在是除夕夜，小黑、无限、哪吒、鹿野正在哪吒豪华大平层里吃年夜饭。
请从以上新闻中选 1-3 条，创意映射为公寓里此刻会发生的具体事件。

映射原则：
- 必须在公寓内物理可发生（断电/手机推送/外卖误送/窗外烟花/轻微震感/电视画面/东西打翻）
- 可大可小，大事件少用（importance≥8 最多 1 个）
- 映射要有创意：加州断电 → 公寓跳闸；股市暴跌 → 哪吒手机弹亏损提醒；地震 → 酒柜里酒瓶叮当响
- memory 字段：所有人能感知到的场景描述，1-2 句，有实体感
- trigger_speaker：第一个开口的角色（小黑/无限/哪吒/鹿野，可留空）

只回复 JSON 数组，不要其他内容：
[
  {{
    "description": "事件简述，10字内",
    "memory": "场景描述，1-2句",
    "importance": 6,
    "trigger_speaker": "哪吒"
  }}
]"""

    msg = await client.messages.create(
        model=_MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    # 替换中文引号为标准双引号
    raw = raw.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    # 提取第一个完整 JSON 数组（防止 LLM 在数组后附加说明文字）
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        raw = m.group(0)

    events: list[dict] = json.loads(raw)
    for ev in events:
        ev["fire_at"] = "immediate"
        ev.setdefault("importance", 6)
        ev.setdefault("trigger_speaker", "")
    return events
