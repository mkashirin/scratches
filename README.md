# Scratches

Scratches is the reverse engineering project inspired by the [Deep Learning 
from Scratch — Building with Python from First Principles by Seth Weidman](http
s://github.com/sadiredwan/books/blob/master/Deep%20Learning%20from%20Scratch%20
-%20Building%20with%20Python%20from%20First%20Principles%20by%20Seth%20Weidman.
pdf). Here You can find pure Python and NumPy implementations of the classic 
machine learning algorithms, such as: K-Nearest Neighbours, Linear and Multiple 
Regressions, elementary and convolutional neural networks.

## Requirements

The only system requirement is to use Conda, Miniconda as Your Python 
package manager.

## Installation and usage

Use the Git CLI to clone this repository into Your working directory 
with the following command:

```
git clone https://github.com/mkashirin/scratches
```

Set up the virlual environment by running the lines below:

```
conda init
conda env create --file environment.yml --name scratches
conda activate scratches
```

Although NumPy is the only robust dependency for the algorithms to work, 
there're also Jupyter, Matplotlib and Pandas in the environment to provide 
plug and play experience.

Also if you want to change the path for the environment, edit the `prefix` 
in the **environment.yml** file (default is `~/anaconda3/envs/scratches`).

After that You can just run the Jupyter sever to access the notebooks from the 
**examples** directory by executing the following command:

```
jupyter lab
```

And that's it, You're all set!

## Suggestions

The only concrete suggestion is not to use it outside of the educational 
domain. 

In case You're not getting something, don't worry. The docstrings 
in the source code can be considered comprehensive documentation. The code is 
also written in a very concise manner and strongly oriented on readability, 
rather than compactness.

Now go ahead and play with machine learning algorithms! Mix and match all the 
avalible structures to build Your own neural networks! Explore the source code 
for deeper understanding of the foundamental ML and AI concepts!
