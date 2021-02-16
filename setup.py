from setuptools import setup, find_packages

setup(
      name='q2-anomaly-detection',
      version='0.0.1',
      description='Anomaly detection for microbiome data',
      long_description=open('README.md').read(),
      license='LICENSE',
      author='George Armstrong',
      author_email='garmstro@eng.ucsd.edu',
      url='https://github.com/gwarmstrong/q2-anomaly-detection',
      packages=find_packages(),
      install_requires=[
            'numpy',
            'scikit-learn',
            'biom-format',
            'matplotlib',
            'seaborn',
            'scipy',
            'pandas',
      ],
      extras_require={
            'analysis': [
                  'jupyter',
                  'emperor',
            ],
            'dev': [
                  'flake8',
            ]
      }
)
