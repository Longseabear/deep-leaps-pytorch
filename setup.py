import setuptools

with open("README.MD", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setuptools.setup(
    name="deep-leaps",
    version="0.0.5",
    author="Leaps",
    author_email="leap1568@gmail.com",
    description="Data driven development based deep learning deepleaps(pytorch)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Longseabear/deep-leaps-pytorch.git",
    packages=setuptools.find_packages(),
    package_data={'deepleaps.app': ['*.yaml'], 'deepleaps': ['workspace','workspace/*', 'workspace/**/*', 'workspace/**/**/*', 'workspace/**/**/**/*']},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements
)
#    download_url='https://github.com/Longseabear/deep-leaps-pytorch/archive/0.0.2.tar.gz',
