# FactOrMisinfo

The goal of this project is to ultimately create a website in which you can enter a URL of an article and be told whether the article is misinformation as well as a possible liberal or conservative bias. 

To accomplish this, we need to use a neural network that can use the text of the articles in tandem with the article tile and outlet that published it to determine whether an article should be considered true or false. However, a neural network doesn't just "read" text, but rather will need a vector to actually understand the article. By using vectorization with Gensim, we can turn each article into a vector, which can be passed into the neural network. 

Below, you can view status updates on the project:

September 25th

After manually compiling 100 articles from websites such as Politifact and Snopes, I was able to run it through a neural network from scikit-learn. As of now, the test accuracy of that network ranges from anywhere between 0.6 and 0.8. While the latter is a decent score, it needs to be more consistent for me to consider it workable, so a more complex neural network will be needed. I've also downloaded the fourth installment of the NELA-GT datasets, NELA-GT-2021, a dataset that contains over 1.8 million articles from 367 different sources. With the immense amount of data that comes with this dataset, we'll be able to train a better model than the one I originally created. 

To-do list as of now:
- Create neural network
- Figure out how to scrape the necessary data from the URL
- Put together the website
- Assemble another neural network that accounts for liberal/conservative bias

Credit to Maurício Gruppi, Benjamin D. Horne, and Sibel Adali for the original code and the NELA-GT-2021 dataset.

[NOTE TO SELF: CITE THE ABOVE PROPERLY]
