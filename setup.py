import setuptools
import glob


def readme():
    with open('README.md') as f:
        return f.read()


scripts = glob.glob('bin/*.sh') + glob.glob('bin/*.py')

setuptools.setup(
    name='wikisim',
    version='1.1.3',
    description='Neural representation of semantic similarity for famous people and places.',
    long_description=readme(),
    long_description_content_type="text/markdown",
    author='Neal Morton',
    author_email='mortonne@gmail.com',
    license='GPLv3',
    url='http://github.com/prestonlab/wikisim',
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    package_data={'wikisim': ['resources/subjects.json']},
    scripts=scripts,
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.7',
    ]
)
