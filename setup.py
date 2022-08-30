from setuptools import setup, find_packages

with open("Readme.md", "r") as fh:
    long_description = fh.read()

setup(name='fhnw-nlp-utils',
      version='0.4.8',
      description='Utilities for NLP courses taught at FHNW.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/markif/fhnw-nlp-utils',
      author='Fabian',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'pandas',
          'psutil',
          'multiprocess',
          'wget',
          'gdown',
          'scikit-learn',
          'nltk',
          'matplotlib',
          'wordcloud',
          'compound-split',
          'pyarrow',
      ],
      zip_safe=False)
