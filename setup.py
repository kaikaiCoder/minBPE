from setuptools import setup, find_packages

# 读取 README 文件内容，用于项目描述
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='minBPE',
    version='0.1.0',
    author='kaikaiCoder',
    author_email='likaikai@stu.xju.edu',
    description='A minimal implementation of Byte Pair Encoding (BPE) for tokenization',
    long_description=long_description,  # 长描述，通常从 README.md 中读取
    long_description_content_type='text/markdown',  # 长描述内容类型
    url='https://github.com/kaikaiCoder/minBPE',  # 项目主页
    packages=find_packages(),  # 自动发现并包含所有 Python 包
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # 依赖的最低 Python 版本
    install_requires=[
        'tiktoken',
        'regex',
    ],
)
