# Copyright (c) 2024 Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import distance
import numpy as np
from bs4 import BeautifulSoup
from collections import deque
from apted.helpers import Tree
from apted import APTED, Config
from rapidfuzz.distance import Levenshtein
from swift.rewards import ORM, orms


def html2matrix(html_str):
    """
    将 html 转 table martix
    Args:
        html_str:

    Returns:

    """
    try:
        # 将 HTML 表格转换为矩阵的函数实现
        soup = BeautifulSoup(html_str, 'html.parser')
        html_str_rows = soup.find_all('tr')
        # 获取表格的行数和列数
        rows = len(html_str_rows)
        cols = 0
        for html_str_row in html_str_rows:
            cell_col_list = html_str_row.contents
            cur_col_num = 0
            col_span = 0
            for cell_col in cell_col_list:
                if not hasattr(cell_col, 'attrs'):
                    continue
                cur_col_num += 1
                cell_attrs = cell_col.attrs
                col_span += int(cell_attrs.get("colspan", 1))
            cols = max(cur_col_num, col_span)

        cell_num = 0
        matrix = np.ones((rows, cols), dtype=int) * -1
        for cur_row, html_str_row in enumerate(html_str_rows):
            cell_col_list = html_str_row.contents
            cur_col = 0
            for cell in cell_col_list:
                if not hasattr(cell, 'attrs'):
                    continue
                cell_attrs = cell.attrs
                rows_over = int(cell_attrs.get("rowspan", 1))
                cols_over = int(cell_attrs.get("colspan", 1))
                is_occupy = np.any(~(matrix[cur_row: cur_row + rows_over, cur_col: cur_col + cols_over] == -1))
                while is_occupy:
                    cur_col += 1
                    is_occupy = np.any(~(matrix[cur_row: cur_row + rows_over, cur_col: cur_col + cols_over] == -1))
                matrix[cur_row: cur_row + rows_over, cur_col: cur_col + cols_over] = cell_num
                cell_num += 1
                cur_col += 1
        return matrix.tolist()
    except:
        return None


class TableTree(Tree):
    def __init__(self, tag, name, colspan=None, rowspan=None, content=None, *children):
        super().__init__(name, *children)
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == "td":
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % (
                self.tag,
                self.colspan,
                self.rowspan,
                self.content,
            )
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.0
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.0


class TEDS:
    """
    基于树编辑距离的相似度计算
    """

    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (
                n_jobs >= 1
        ), "n_jobs must be an integer greater than 1"
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        """Tokenizes table cells"""
        self.__tokens__.append("<%s>" % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != "unk":
            self.__tokens__.append("</%s>" % node.tag)
        if node.tag != "td" and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        """Converts HTML tree to the format required by apted"""
        global __tokens__
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
                new_node = TableTree(
                    node.tag,
                    int(node.attrib.get("colspan", "1")),
                    int(node.attrib.get("rowspan", "1")),
                    cell,
                    *deque(),
                )
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        """Computes TEDS score between the prediction and the ground truth of a
        given sample
        """
        # try_import("lxml")
        from lxml import etree, html

        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        if pred.xpath("//table") and true.xpath("//table"):
            pred = pred.xpath("//table")[0]
            true = true.xpath("//table")[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(
                tree_pred, tree_true, CustomConfig()
            ).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0


def extract_html_tables_bs4(markdown_text):
    """
    使用BeautifulSoup从Markdown字符串中提取HTML表格

    参数:
        markdown_text: 包含HTML表格的Markdown字符串

    返回:
        所有HTML表格字符串的列表
    """
    # 将Markdown转换为BeautifulSoup对象
    soup = BeautifulSoup(markdown_text, 'html.parser')

    # 查找所有table标签
    tables = soup.find_all('table')

    # 返回表格的字符串表示
    return [str(table) for table in tables]


teds = TEDS(n_jobs=4)


def reward_func(pred_markdown, gt_markdown):
    """
    奖励函数计算
    Args:
        pred_markdown:
        gt_markdown:

    Returns:

    """
    try:
        gt_tables = extract_html_tables_bs4(gt_markdown)
        pred_tables = extract_html_tables_bs4(pred_markdown)
        # 如果 pred_markdown 和 gt_markdown 中表格数量不等，直接使用编辑距离相似度作为 reward
        if len(gt_tables) != len(pred_tables):
            reward = Levenshtein.normalized_similarity(pred_markdown, gt_markdown)
        # pred_markdown 和 gt_markdown 中表格数量相等，依次计算二者表格之间的 teds，以及去除表格剩余字符串的编辑距离相似度
        # 每个表格和剩余文本占据相同的权重
        else:
            sub_reward_list = []
            pred_remain_markdown = pred_markdown
            gt_remain_markdown = gt_markdown
            for gt_table, pred_table in zip(gt_tables, pred_tables):
                pred_remain_markdown = pred_remain_markdown.replace(pred_table, "")
                gt_remain_markdown = gt_remain_markdown.replace(gt_table, "")
                # gt 可能存在问题，导致无法转 matrix，这里增加计算鲁棒性
                # 如果筛选数据集 gt 中的 table html 结构都正常，该判断不会触发
                gt_matrix = html2matrix(gt_table)
                if gt_matrix is None:
                    sub_reward_list.append(0.0)
                    continue
                pred_matrix = html2matrix(pred_markdown)
                # pred 无法转 matrix，当前表格 reward=0
                if pred_matrix is None:
                    sub_reward_list.append(0.0)
                    continue
                if pred_matrix == gt_matrix:
                    sub_reward_list.append(1.0)
                elif np.any(pred_matrix == -1):
                    sub_reward_list.append(0.0)
                else:
                    sub_reward_list.append(teds.evaluate(gt_table, pred_table))
            sub_reward_list.append(Levenshtein.normalized_similarity(pred_remain_markdown, gt_remain_markdown))
            reward = sum(sub_reward_list) / len(sub_reward_list)
    except:
        # try 异常，表明 html 结构异常，直接返回 0.0
        reward = 0.0
    return reward


class TableQianfanOCRDistance(ORM):

    def __call__(self, completions, **kwargs):
        solution = kwargs['solution']
        rewards = []
        for pred, gt in zip(completions, solution):
            reward = reward_func(pred, gt)
            rewards.append(reward)
        return rewards


orms['external_qianfanocr_table'] = TableQianfanOCRDistance

if __name__ == '__main__':
    pred_markdown, gt_markdown = "123", "456"
    reward = reward_func(pred_markdown, gt_markdown)
    print(reward)
