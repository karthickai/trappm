from setuptools import setup, find_packages

setup(
    name='trappm',
    version='0.1.0',
    author='Karthick Panner Selvam',
    author_email='karthick.pannerselvam@uni.lu',
    description='TraPPM Performance Prediction Model',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='http://github.com/karthickai/trappm',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.10',
    install_requires=[
        'torch==2.0.0',
        'torch-geometric==2.3.0',
        'torch-cluster==1.6.1',
        'torch-scatter==2.1.1',
        'torch-sparse==0.6.17',
        'torch-spline-conv==1.2.2',
        'timm==0.6.13',
        'onnx==1.13.1',
        'networkx==3.1',
        'rich'
    ],
    entry_points={
        'console_scripts': [
        ],
    },
)
