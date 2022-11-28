# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/5/20 21:37
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/6/16 2:03


# def merge_field_content(
#         field: list,
#         content: list,
#         separator: str
# ) -> str:
#     """
#
#     Merge fields and corresponding content, used in SQL statements.
#
#     Args:
#         field (list): List of field names.
#         content (list): List of contents, corresponding to the fields.
#         separator (str): Separator between different field contents, usually ',' or 'and'.
#
#     Returns:
#         Str: The resulting string after merging, which can be added to the SQL statement.
#
#     """
#
#     # Initialize.
#     return_word = ""
#
#     # Realize using loop
#     for item in range(len(field)):
#         # Judge whether the content is a string.
#         # The string needs to be additionally quoted
#         if type(content[item]) == type('str'):
#             return_word += "%s= '%s'" % (field[item], content[item])
#
#         else:
#             return_word += "%s= %s" % (field[item], content[item])
#
#         # Judge whether it is the last item.
#         if item != (len(field) - 1):
#             return_word += " %s " % (separator)
#
#     return return_word


# def merge_field_or_content(
#         list: list,
#         separator: str,
#         content: bool = True
# ) -> str:
#     """
#
#     Merge fields or contents, used in SQL statements.
#
#     Args:
#         list (list): List of items that will be merged.
#         separator (str): Separator between different field contents, usually ',' or 'and'.
#         content (bool): Whether the entered item is content. Default as True.
#
#     Returns:
#         Str: The resulting string after merging, which can be added to the SQL statement.
#
#     """
#
#     # Initialize.
#     return_word = ""
#
#     # Realize using loop
#     for item in range(len(list)):
#         # Judge whether the content is a string.
#         # The string needs to be additionally quoted when the entered item is content.
#         if content and type(list[item]) == type('str'):
#             return_word += "'%s'" % (list[item])
#
#         else:
#             return_word += "%s" % (list[item])
#
#         # Judge whether it is the last item.
#         if item != (len(list) - 1):
#             return_word += " %s " % (separator)
#
#     return return_word
