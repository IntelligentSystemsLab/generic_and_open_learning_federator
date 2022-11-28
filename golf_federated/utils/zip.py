# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/5/25 10:32
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/6/16 2:16

# import zipfile


# def zip_model(
#         model_path: str,
#         zip_path: str
# ) -> None:
#     """
#
#     Save the model parameter file as a zip file.
#
#     Args:
#         model_path (str): Path to the model parameter file.
#         zip_path (str): Path to zip file.
#
#     """
#
#     # Open the specified zip file.
#     first_model_zipper = zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED)
#
#     # Write the model parameters.
#     first_model_zipper.write(filename=model_path)
#
#     # Close the zip file.
#     first_model_zipper.close()


# def read_zip(zip_path: str) -> bytes:
#     """
#
#     Read the contents of a zip file.
#
#     Args:
#         zip_path (str): Path to zip file.
#
#     Returns:
#         Bytes: Contents of the zip file.
#
#     """
#
#     # Open the specified zip file.
#     binfile_first = open(zip_path, 'rb')
#
#     # Read the contents.
#     bin_infomation = binfile_first.read()
#
#     return bin_infomation
