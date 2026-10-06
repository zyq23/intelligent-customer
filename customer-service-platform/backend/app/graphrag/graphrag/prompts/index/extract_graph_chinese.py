# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Chinese version of GraphRAG extraction prompt."""

GRAPH_EXTRACTION_PROMPT = """
-目标-
给定一段可能是相关活动的文本文档和一个实体类型列表，从文本中识别所有这些类型的所有实体以及所有已识别实体之间的关系。

-步骤-
1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- entity_name: 实体名称，大写
- entity_type: 以下类型之一：[{entity_types}]
- entity_description: 实体的属性和活动全面描述
使用以下格式将每个实体格式化为 ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 从步骤 1 识别的实体中，识别所有明显相关的 (source_entity, target_entity) 对。
对于每对相关实体，提取以下信息：
- source_entity: 源实体名称，如步骤 1 所识别
- target_entity: 目标实体名称，如步骤 1 所识别
- relationship_description: 解释你认为源实体和目标实体为什么相互关联
- relationship_strength: 一个数字分数，表示源实体和目标实体之间关系的强度
使用以下格式将每个关系格式化为 ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. 以中文返回所有实体和关系的单个列表。使用 **{record_delimiter}** 作为列表分隔符。

4. 完成后，输出 {completion_delimiter}

######################
-示例-
######################
示例 1:
Entity_types: 组织，人物
文本:
 Verdantis 中央机构定于周一和周四开会，该机构计划在周四太平洋夏令时间下午 1:30 发布最新政策决定，随后召开新闻发布会，中央机构主席 Martin Smith 将回答问题。投资者预计市场策略委员会将把基准利率维持在 3.5%-3.75% 区间。
######################
输出:
("entity"{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}组织{tuple_delimiter}中央机构是 Verdantis 的联邦储备局，正在周一和周四设定利率){record_delimiter}
("entity"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}人物{tuple_delimiter}Martin Smith 是中央机构的主席){record_delimiter}
("entity"{tuple_delimiter}MARKET STRATEGY COMMITTEE{tuple_delimiter}组织{tuple_delimiter}中央机构委员会制定利率政策和货币增长的关键决策){record_delimiter}
("relationship"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}Martin Smith 是中央机构主席并将回答新闻发布会问题{tuple_delimiter}9){completion_delimiter}

######################
示例 2:
Entity_types: 组织
文本:
TechGlobal (TG) 股票在全球交易所周四上市首日暴涨。但 IPO 专家警告称，这家半导体公司在公开市场的上市并不代表其他新上市公司可能的表现。

TechGlobal 曾是上市公司，2014 年被 Vision Holdings 私有化。这家资深芯片设计商表示其 powering 85% 的高端智能手机。
######################
输出:
("entity"{tuple_delimiter}TECHGLOBAL{tuple_delimiter}组织{tuple_delimiter}TechGlobal 是在全球交易所上市的股票，powering 85% 高端智能手机){record_delimiter}
("entity"{tuple_delimiter}VISION HOLDINGS{tuple_delimiter}组织{tuple_delimiter}Vision Holdings 是一家曾拥有 TechGlobal 的公司){record_delimiter}
("relationship"{tuple_delimiter}TECHGLOBAL{tuple_delimiter}VISION HOLDINGS{tuple_delimiter}Vision Holdings 曾于 2014 年至现在拥有 TechGlobal{tuple_delimiter}5){completion_delimiter}

######################
-真实数据-
######################
Entity_types: {entity_types}
文本：{input_text}
######################
输出:"""

CONTINUE_PROMPT = "上一次提取中遗漏了许多实体和关系。记住只发出与之前提取的任何类型匹配的实体。用相同的格式在下面添加："

LOOP_PROMPT = "看起来一些实体和关系可能仍然遗漏。回答 Y 或 N 如果仍有需要添加的实体或关系。\n"
