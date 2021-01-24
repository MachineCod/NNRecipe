from setuptools import setup, find_packages

setup(
    name='nn_recipe',
    version='0.1',
    description='Deep learning framework made for educational purposes',
    url='http://github.com/mgtm98/NNRecipe',
    author='Mohamed Gamal - Mohamed Adel - Ahmed Kaled - Mahmoud Hassan - Mohamed Abduallah - Mariem Abdelrahman - Yasmin Alaa',
    author_email='mgtmprog@gmail.com',
    license='GNU GPLv3',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License ::  GNU GPLv3 License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'numpy', 'pillow', 'setuptools'
    ],
    python_requires='>=3.7',
)