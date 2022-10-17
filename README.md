# FactOrMisinfo

The goal of this project is to ultimately create a website in which you can enter a URL of an article and be told whether the article is misinformation as well as a possible liberal or conservative bias. 

To accomplish this, we need to use a neural network that can use the text of the articles in tandem with the article tile and outlet that published it to determine whether an article should be considered true or false. However, a neural network doesn't just "read" text, but rather will need a vector to actually understand the article. By using vectorization with Gensim, we can turn each article into a vector, which can be passed into the neural network. 

A neural network is a method used in machine learning in which a computer learns how to perform a task based on information fed to it. A neural network is made up of many layers, with each layer made up of neurons. Each neuron contains an "activation", which is a number between 0 and 1. When the neural network is asked to analyze something, the first layer of neurons gets their activations, and the pattern of activations in the first layer will lead to a certain pattern of activations in the next layer, and so on and so forth until the final layer, in which those final activations will be used to determine an outcome. For example, that final layer in this project should only have two neurons: one which would classify an article as misinformation, and another that would classify it as factual. Whichever one of these neurons has the higher activation determines what the article is classified as.

[Explain what it actually means to turn something into a vector]

## Project Journal

### September 25th, 2022:

After manually compiling 100 articles from websites such as Politifact and Snopes, I was able to run it through a neural network from scikit-learn. As of now, the test accuracy of that network ranges from anywhere between 0.6 and 0.8. While the latter is a decent score, it needs to be more consistent for me to consider it usable, so a more complex neural network will be needed.

### October 5th, 2022

A new model has been created! Using Keras, I've been able to create a new language classifier while still utilizing Gensim vectorization (although the goal is to eventually give the job of vectorization to BERT), and the classifier currently has a consistent accuracy of 80% whilst utilizing 30 epochs, trained on the information from the dataset I've manually created. While this is great, it'd still be preferable to get the accuracy a bit higher, so by using the larger dataset, I'll be able to train the model more accurately. 

At this point, I'll have to create the website relatively soon (hopefully by the end of October?) as to make the model usable by the general populace. All in all, I can say that I'm happy with the progress being made. I'll update this when I either finish a working website or get the model running on the larger dataset.

### October 13th, 2022

The BERT model has successfully been implemented, replacing the Gensim vectorization used previously. Comparing the two, the Gensim vectorization consistently achieved an accuracy of 80% on the test set. However, when BERT was put in place, the accuracy varied a bit more, with 80% being the low and having a high of 95%. While not 100%, 95% is nothing to sneeze at. In addition, BERT is much faster, cutting down the time it takes to run from about two and a half minutes to around thirty seconds. Using the faster BERT model will make this process much simpler for future testing. 

I've also implemented a piece of code that allows you to write your own statement to test. Unfortunately, this doesn't seem to be too accurate, with it classifying the statement "Barack Obama was born in Africa and is a Muslim terrorist" as true. At this point, the only thing that can really be done to make this model more accurate is to get a larger dataset. Hopefully I'll be able to get a website up and running soon, so that this model can actually be used by the public.

## To-do list as of now:
- Put together the website
- Train neural network on larger dataset
- Assemble another neural network that accounts for liberal/conservative bias
- Figure out how to scrape the necessary data from the URL

## Background

I'm someone who's always been intrigued by politics and computer science, but the two paths never really felt "mergeable" to me. Within my interest in politics, I wanted to focus on reducing partisan polarization within our communities after watching the insurrection on January 6th in real time. In my work, I created a committee at my high school dedicated to facilitating conversations between people of opposing viewpoints, and covered topics like reading bills yourself instead of trusting what the media tells you, and how different news outlets will give you the "facts" but purposefully leave out or include information so that you aren't really arriving at your own conclusion. Through this committee, I started realizing that a lot of the reason why we're moving towards the extremes is due to rampant misinformation spreading throughout communities. I've wanted to figure out some way to reduce people from being fooled, so I decided to take what I was learning in my machine learning course and apply it here. This lead to the creation of FactOrMisinfo (working title), which is dedicated to stopping United States politics be defined by bad actors.


Credit to Maurício Gruppi, Benjamin D. Horne, and Sibel Adali for the original code and the NELA-GT-2021 dataset.

[NOTE TO SELF: CITE THE ABOVE PROPERLY]
