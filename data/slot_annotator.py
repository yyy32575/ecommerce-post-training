"""
槽位标注模块

识别并标注文本中的敏感槽位，包括价格、日期、库存状态、促销信息、品牌名等。
使用正则表达式 + 规则混合方法。
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 槽位正则规则定义
SLOT_PATTERNS = {
    "price": [
        r"[¥￥]\s*\d+(?:[,，]\d{3})*(?:\.\d{1,2})?",  # ¥199, ¥1,299.00
        r"\d+(?:[,，]\d{3})*(?:\.\d{1,2})?\s*元",      # 199元, 1299.00元
        r"\d+(?:\.\d{1,2})?\s*块",                       # 99.9块
        r"售价[：:]\s*[¥￥]?\d+(?:\.\d{1,2})?",          # 售价：199
        r"价格[：:]\s*[¥￥]?\d+(?:\.\d{1,2})?",          # 价格：299
    ],
    "date": [
        r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?",     # 2024-01-01, 2024年1月1日
        r"\d{1,2}[-/月]\d{1,2}[日号]",                  # 1月1日, 01/01
        r"(?:今天|明天|后天|昨天|前天)",                  # 今天、明天
        r"(?:本周|下周|上周)[一二三四五六日天末]?",        # 本周末
        r"\d+[天周月年](?:内|后|前)",                     # 3天内、2周后
        r"(?:元旦|春节|五一|国庆|双十一|618|双12)",       # 节假日
    ],
    "stock": [
        r"(?:现货|有货|无货|缺货|断货|库存不足|备货中|预售)",  # 库存状态词
        r"库存[：:只件个]\s*\d+",                           # 库存：99件
        r"还剩\s*\d+\s*[只件个]",                           # 还剩5件
        r"(?:即将售罄|快抢完了|最后\d+件)",                 # 紧迫度
        r"(?:到货|补货)(?:时间)?[：:]\s*[\d一二三四五六七八九十]+[天周月]?", # 到货时间
    ],
    "promotion": [
        r"(?:满\d+减\d+|满减|折扣|优惠|打折|特价|秒杀|闪购|限时)",  # 促销关键词
        r"\d+(?:\.\d)?\s*折",                              # 8折、8.5折
        r"(?:立减|直降|优惠|减)\s*[¥￥]?\d+(?:\.\d{1,2})?", # 立减50
        r"(?:买\d+(?:送|赠)\d+|买[一二三四五](?:送|赠)[一二三四五])", # 买一送一
        r"(?:优惠券|折扣码|coupon)[：:]\s*\w+",            # 优惠券代码
        r"(?:活动|促销|大促)[截止到]?\s*\d{1,2}[月/]\d{1,2}[日号]?", # 活动截止
    ],
    "brand": [
        r"(?:华为|小米|苹果|三星|OPPO|vivo|荣耀|一加|索尼|联想|戴尔|惠普|华硕|宏碁)",
        r"(?:Nike|Adidas|New Balance|李宁|安踏|特步|361度|匹克|鸿星尔克)",
        r"(?:海尔|美的|格力|西门子|博世|松下|飞利浦|LG|三菱|大金)",
        r"(?:宝洁|联合利华|欧莱雅|资生堂|雅诗兰黛|SK-II|兰蔻|香奈儿|迪奥)",
        r"(?:阿迪达斯|耐克|优衣库|ZARA|H&M|Gap|levi's|Levis)",
    ],
}


class SlotAnnotator:
    """
    文本槽位标注器。

    识别并标注文本中的敏感信息槽位（价格、日期、库存、促销、品牌），
    使用 XML 风格标签标注：<slot type="price">¥199</slot>

    Args:
        slot_types: 需要标注的槽位类型列表，默认标注所有类型
        text_fields: 需要处理的文本字段列表，默认 ["text", "response"]
        keep_original: 是否在输出中保留原始文本，默认 True
    """

    def __init__(
        self,
        slot_types: Optional[List[str]] = None,
        text_fields: Optional[List[str]] = None,
        keep_original: bool = True,
    ) -> None:
        self.slot_types = slot_types or list(SLOT_PATTERNS.keys())
        self.text_fields = text_fields or ["text", "response"]
        self.keep_original = keep_original
        # 预编译正则
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for slot_type in self.slot_types:
            patterns = SLOT_PATTERNS.get(slot_type, [])
            self._compiled_patterns[slot_type] = [
                re.compile(p) for p in patterns
            ]
        logger.info("初始化 SlotAnnotator: slot_types=%s", self.slot_types)

    def _annotate_text(self, text: str) -> Tuple[str, Dict[str, List[str]]]:
        """
        对单条文本进行槽位标注。

        Returns:
            Tuple[str, Dict]: (标注后文本, 各槽位类型的匹配值字典)
        """
        annotated = text
        found_slots: Dict[str, List[str]] = {t: [] for t in self.slot_types}

        # 收集所有匹配（带位置），避免嵌套替换错误
        matches: List[Tuple[int, int, str, str]] = []
        for slot_type in self.slot_types:
            for pattern in self._compiled_patterns[slot_type]:
                for m in pattern.finditer(text):
                    matches.append((m.start(), m.end(), slot_type, m.group()))

        # 按位置排序，处理重叠（保留最长匹配）
        matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        non_overlapping: List[Tuple[int, int, str, str]] = []
        last_end = 0
        for start, end, slot_type, value in matches:
            if start >= last_end:
                non_overlapping.append((start, end, slot_type, value))
                found_slots[slot_type].append(value)
                last_end = end

        # 从后向前替换，避免位移问题
        for start, end, slot_type, value in reversed(non_overlapping):
            tag = f'<slot type="{slot_type}">{value}</slot>'
            annotated = annotated[:start] + tag + annotated[end:]

        return annotated, found_slots

    def annotate_single(self, item: Dict) -> Dict:
        """
        对单条数据进行槽位标注。

        Args:
            item: 包含文本字段的字典

        Returns:
            Dict: 标注后的数据字典
        """
        result = dict(item)
        all_slots: Dict[str, List[str]] = {t: [] for t in self.slot_types}

        for field in self.text_fields:
            text = item.get(field, "")
            if not text:
                continue
            annotated_text, found_slots = self._annotate_text(text)
            if self.keep_original:
                result[f"{field}_original"] = text
            result[field] = annotated_text
            for slot_type, values in found_slots.items():
                all_slots[slot_type].extend(values)

        result["slots"] = {k: list(set(v)) for k, v in all_slots.items() if v}
        return result

    def annotate(self, dataset: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        对数据集进行批量槽位标注。

        Args:
            dataset: 包含文本字段的字典列表

        Returns:
            Tuple[List[Dict], Dict]: (标注后数据集, 标注统计信息)
        """
        logger.info("开始槽位标注，数据量: %d", len(dataset))

        annotated_dataset: List[Dict] = []
        slot_counts: Dict[str, int] = {t: 0 for t in self.slot_types}
        items_with_slots = 0

        for idx, item in enumerate(dataset):
            annotated_item = self.annotate_single(item)
            annotated_dataset.append(annotated_item)

            slots = annotated_item.get("slots", {})
            if slots:
                items_with_slots += 1
                for slot_type, values in slots.items():
                    slot_counts[slot_type] = slot_counts.get(slot_type, 0) + len(values)

            if (idx + 1) % 1000 == 0:
                logger.info("已标注 %d / %d 条数据", idx + 1, len(dataset))

        stats = {
            "total": len(dataset),
            "items_with_slots": items_with_slots,
            "slot_counts": slot_counts,
        }
        logger.info(
            "槽位标注完成: 总计=%d, 含槽位=%d, 槽位分布=%s",
            stats["total"],
            stats["items_with_slots"],
            slot_counts,
        )
        return annotated_dataset, stats
