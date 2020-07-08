import setuptools

setuptools.setup(
    name="pyebm",
    version="2.0.2",
    author="Vikram Venkatraghavan",
    author_email="v.venkatraghavan@erasmusmc.nl",
    description="Toolbox for event-based modeling",
    long_description="Contains codes for EBM and DEBM with several options for mixture modeling",
    long_description_content_type="text/markdown",
    url="https://github.com/88vikram/pyebm/",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
      'numpy',
      'pandas',
      'scikit-learn',
      'scipy',
      'six',
      'statsmodels',
      'matplotlib',
      'seaborn',
      'requests',
      'tqdm'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
)
