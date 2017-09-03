# how_to_win_slot_machines
This is my code for the code challenge "How to Do Win Slot Machines - Intro to Deep Learning #13' by Siraj Raval on YouTube

# Coding Challenge - Due Date - Thursday, April 13th at 12 PM PST

The coding challenge for this video is to use multiple slot machines instead of one. This way, state is taken into account. See this [article](https://getstream.io/blog/introduction-contextual-bandits/) for more info on this. Bonus points given for applying the code to a real world use case. You'll learn more about how policy and value functions are related in reinforcement learning by doing this exercise. 

## Overview

I extrapolated the idea of slot machines to the stock market. It's basically a slot machine for rich folks, amirite? I built several different trading 'bandit bots' which each have a strategy that they follow. Since the 'brains' of the bots are irregular (compared to a tensor of data for example), they do not lend themselves easily to neural networks. Reinforcement learning is great for this, because it does not require a loss function, only some reward criteria. We can test each of the bots and using the multi-armed bandit algorithm, determine which is the best-performing. 

## Dependencies

* tensorflow (https://www.tensorflow.org/install/)
* numpy
* pandas - for loading in stock data
* matplotlib - for graphhing

## Usage

Run `jupyter notebook` in the main directory of this repository in terminal to see the code pop up in your browser. 

Install jupyter [here](http://jupyter.readthedocs.io/en/latest/install.html) if you haven't already.

## Credits

The credits for the base code go to [awjuliani](https://github.com/awjuliani). Siraj added a wrapper to get people started. I built the bots and the updated loss function. 

