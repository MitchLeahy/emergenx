from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

with open('HISTORY.md') as history_file:
    HISTORY = history_file.read()

setup_args = dict(
    name='Emergenx',
    version='0.1.0',
    description='Useful tools to work with NN in Python',
    long_description_content_type="text/markdown",
    long_description=README + '\n\n' + HISTORY,
    license='MIT',
    packages=find_packages(),
    author='Mitch Leahy',
    author_email='leahy.mi@northeastern.edu',
    keywords=['emergence'],
    url='https://github.com/MitchLeahy/emergenx',
    download_url='https://pypi.org/project/emergenx/'
)

install_requires = [
    'random',
    'altair',
    'numpy',
    'pandas',
    'sklearn',
    'io',
    'base64',
    'PIL >= 7.0.0'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)