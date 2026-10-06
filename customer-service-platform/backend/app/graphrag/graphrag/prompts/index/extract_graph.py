# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A file containing prompts definition - Chinese version for Chinese text processing."""

GRAPH_EXTRACTION_PROMPT = """
-目标-
给定一段可能是相关活动的文本文档和一个实体类型列表，从文本中识别所有这些类型的所有实体以及所有已识别实体之间的关系。

-步骤-
1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- entity_name: 实体名称
- entity_type: 以下类型之一：[{entity_types}]
- entity_description: 实体的属性和活动全面描述
格式化为 ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 从步骤 1 识别的实体中，识别所有明显相关的 (source_entity, target_entity) 对。
对于每对相关实体，提取以下信息：
- source_entity: 源实体名称，如步骤 1 所识别
- target_entity: 目标实体名称，如步骤 1 所识别
- relationship_description: 解释为什么源实体和目标实体相互关联
- relationship_strength: 1-10 的数字分数表示关系强度
格式化为 ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. 以中文返回所有实体和关系的列表。使用 **{record_delimiter}** 作为列表分隔符。

4. 完成后，输出 {completion_delimiter}

-真实数据-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""

CONTINUE_PROMPT = "上一次提取遗漏了很多实体和关系。请记住只发射与之前提取的类型匹配的实体。在下面添加：\n"
LOOP_PROMPT = "看起来仍可能遗漏了一些实体和关系。回答 Y 或 N 如果仍有需要添加的实体或关系。\n"
