# FactOrMisinfo

## Background

I'm someone who's always been intrigued by politics and computer science, but the two paths never really felt "mergeable" to me. Within my interest in politics, I wanted to focus on reducing partisan polarization within our communities after watching the insurrection on January 6th in real time. In my work, I created a committee at my high school dedicated to facilitating conversations between people of opposing viewpoints, and covered topics like reading bills yourself instead of trusting what the media tells you, and how different news outlets will give you the "facts" but purposefully leave out or include information so that you aren't really arriving at your own conclusion. Through this committee, I started realizing that a lot of the reason why we're moving towards the extremes is due to rampant misinformation spreading throughout communities. I've wanted to figure out some way to reduce people from being fooled, so I decided to take what I was learning in my machine learning course and apply it here. This lead to the creation of FactOrMisinfo (working title), which is dedicated to stopping United States politics be defined by bad actors.

## Mission

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

### October 18th, 2022
I've finally got some of the scraping to work now! Using BeautifulSoup, I've been able to successfully scrape articles from the Associated Press, CNN, and FOX News and run them through the model! However, this has also shown some of the issues with the model again. Technically, all these sources should be "true", just with their own bias on things, which is what the next neural network would account for. However, the model is a bit iffy with all of these. While it was very confident that the CNN article was true, it assumed the FOX News article to be false. For the AP, it generally assumed articles to be true, but would sometimes classify it with false, even though the AP should be the least biased of the three. There are two ways I see to fix this: one, either alter the scraping to see if it's picking up something it shouldn't, or two, do what I previously suggested and just get a larger dataset. From what I've seen, I think the latter would fix the most things possible.

### October 20th, 2022
The model can now scrape from a variety of websites, from MSNBC to InfoWars. On the bad side, this has exposed the model's weakness more than ever. The model rarely gets these websites correct, although it had a 95% accuracy on the training data. I think the only thing that can be done now is to just get more articles and run the larger dataset through the model.

### January 18th, 2023
It's been a while since the last update, mostly since due to work on the website. Unfortunately, I've recently learned that I have no clue how to do this. As of now, I have a working frontend, but I can't figure out how to make it call my Python program. Once I figure that out, I'll update this. Hopefully it'll be before the end of this month.

As a side note, GitHub doesn't allow you to upload more than 100 files at a time, which my website so far is, so I'll figure that out later.

## To-do list as of now:
- Put together the website
- Train neural network on larger dataset
- Assemble another neural network that accounts for liberal/conservative bias
- Create scraping solutions for each website until a better option can be found

## Credits
### Libraries Used
- scikit-learn
- pandas
- keras
- numpy
- transformers
- PyTorch
- BeautifulSoup4
- requests

### Special Thanks
Special thanks to the people who helped me with various parts of the project. Thank you to Alex Calderwell for getting me started on the project and getting me to a place where I could do projects like this 1000 times over. Thank you to Maurício Gruppi who kindly responded to my inquiry on his project which had a similar goal and how we went about completing it.
