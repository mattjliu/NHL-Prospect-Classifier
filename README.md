# NHL-Prospect-Classifier

This is a code dump for a text-classifier for NHL prospects. The intent was to train a model on scouting reports to predict the probability of success of prospects.

## Background

The scouting reports were scraped from [this mock-draft site](https://www.draftsite.com/nhl/mock-draft/2019/) which contains short scouting reports for most players on their respective pages, dating back to the draft class of 2011 (about 1500 scouting reports). NHL player stats were scraped from hockeydb [here](http://www.hockeydb.com/ihdb/draft/index.html).

The mean of games played per draft class was used to discriminate between the positive and negative class. For example, a given player drafted in 2015 would have to play more games in the NHL than the average number of games played by players drafted in 2015 (for those whose reports are available to me) in order to be in the positive class. All other players would fall in the negative class.

Furthermore, any player drafted in 2016 or later could only be placed in the positive class, since it was "too early to tell". Otherwise, they would be placed in the holdout/validation set along with the 2019 draft class.

Finally, all players with 1 sentence scouting reports were removed from the final modelling set since they would most likely be unknown players.

## Training the Model

The model used was a simple bag-of-words model. Logistic regression was selected as the classifier given its probabilistic nature. Stop words were also filtered out and tf-idf was applied before training the model. 

Because of lack-of-data (only 829 reports), a proper test set was not created. Instead, the model was simply cross-validated on the train set and predictions were produced on the validation set (the aformentioned "too early to tell" set of players).

Here are some (expectedly poor) performance metrics of the model:
* Average cross validation ROC AUC: **0.6839**
* Average cross validation accuracy: **0.6545**
* Accuracy on training data: **0.9143**

Here is the list of the 20 most important positive and negative words (by order) for the model:

```sh
================= Important Positive Features =================
  1:nhl                  T: 59 F: 47
  2:player               T:139 F:162
  3:end                  T:142 F:210
  4:hockey               T: 52 F: 43
  5:scoring              T: 53 F: 43
  6:high                 T: 79 F: 78
  7:linemates            T: 18 F:  5
  8:best                 T: 29 F: 20
  9:chances              T: 30 F: 22
 10:defenders            T: 36 F: 24
 11:line                 T: 61 F: 58
 12:net                  T: 40 F: 56
 13:early                T: 34 F: 24
 14:pro                  T:176 F:315
 15:class                T: 20 F:  5
 16:doesn                T: 38 F: 35
 17:scorer               T: 41 F: 30
 18:bigger               T: 39 F: 43
 19:vision               T: 62 F: 51
 20:space                T: 24 F: 17
 
 ================= Important Negative Features =================
  1:solid                T: 30 F: 84
  2:good                 T:159 F:308
  3:nice                 T: 19 F: 48
  4:projects             T: 17 F: 41
  5:term                 T: 51 F:112
  6:school               T:  2 F: 17
  7:prospect             T: 43 F:102
  8:situations           T:  9 F: 21
  9:committed            T: 23 F: 60
 10:glove                T:  1 F: 22
 11:simple               T:  1 F: 13
 12:athleticism          T:  0 F: 13
 13:powerful             T: 10 F: 20
 14:real                 T: 33 F: 52
 15:frame                T: 18 F: 41
 16:tough                T: 20 F: 48
 17:growing              T:  5 F: 17
 18:style                T: 13 F: 34
 19:university           T: 17 F: 49
 20:game                 T:117 F:219

```
The rather rudimentary nature of model definitely shows in the importance for some of these features.

## Predictions

Of course, none of this would be any fun if we didn't try to predict the future, so here are the model's most confident predictions for positive class players on the 2019 draft class:
1. [Kirby Dach](https://www.draftsite.com/nhl/player/kirby-dach/29273/)**,**  ***74.83%***
2. [Dylan Cozens](https://www.draftsite.com/nhl/player/dylan-cozens/30041/)**,**  ***73.64%***
3. [Alexander Newhook](https://www.draftsite.com/nhl/player/alexander-newhook/29318/)**,**  ***72.39%***
4. [Alex Turcotte](https://www.draftsite.com/nhl/player/alex-turcotte/29275/)**,**  ***71.93%***
5. [Philip Broberg](https://www.draftsite.com/nhl/player/philip-broberg/32317/)**,**  ***70.46%***
6. [Robert Mastrosimone](https://www.draftsite.com/nhl/player/robert-mastrosimone/32225/)**,**  ***69.96%***
7. [Kaapo Kakko](https://www.draftsite.com/nhl/player/kaapo-kakko/30042/)**,**  ***69.05%***
8. [Peyton Krebs](https://www.draftsite.com/nhl/player/peyton-krebs/29276/)**,**  ***68.79%***
9. [Jack Hughes](https://www.draftsite.com/nhl/player/jack-hughes/29274/)**,**  ***68.42%***
10. [Patrik Puistola](https://www.draftsite.com/nhl/player/patrik-puistola/32373/)**,**  ***68.12%***

and the following are the model's most confident predictions for positive class players in the whole validation set:
1. [Vitaly Abramov](https://www.draftsite.com/nhl/player/vitali-abramov/23471/)**,**  ***82.25%***
2. [Joel Farabee](https://www.draftsite.com/nhl/player/joel-farabee/28715/)**,**  ***81.17%***
3. [Timothy Liljegren](https://www.draftsite.com/nhl/player/timothy-liljegren/25845/)**,**  ***80.01%***
4. [Barrett Hayton](https://www.draftsite.com/nhl/player/barrett-hayton/28683/)**,**  ***79.40%***
5. [Jonatan Berggren](https://www.draftsite.com/nhl/player/jonatan-berggren%C2%A0/29262/)**,**  ***77.55%***
6. [Kirby Dach](https://www.draftsite.com/nhl/player/kirby-dach/29273/)**,**  ***74.83%***
7. [Mario Ferraro](https://www.draftsite.com/nhl/player/mario-ferraro/27957/)**,**  ***74.67%***
8. [Dmitry Sokolov](https://www.draftsite.com/nhl/player/dimitri-sokolov/22459/)**,**  ***74.44%***
9. [Rasmus Asplund](https://www.draftsite.com/nhl/player/rasmus-asplund/22945/)**,**  ***74.05%***
10. [Joe Veleno](https://www.draftsite.com/nhl/player/joe-veleno/26377/)**,**  ***73.96%***

## Data and Results

You can find all predictions for the long report model in [predictions/long/](https://github.com/mattjliu/NHL-Prospect-Classifier/tree/master/predictions/long). It also contains all of the models classifications on the train set.

Scatter plots describing the behaviour of the model can be found in [plots/](https://github.com/mattjliu/NHL-Prospect-Classifier/tree/master/plots).

The scraped (and merged) data can be found in [data/merged/](https://github.com/mattjliu/NHL-Prospect-Classifier/tree/master/data/merged).

## Further Reading

A paper on this specific subject was written in 2017 by a hockey analytics firm. Their analysis combined both prospect performance statistics and scouting reports to predict a player's future AHL performance. The quality and quantity of scouting report data used was also substantially better.

You can read the paper [here](https://pdfs.semanticscholar.org/2f0a/a4de57e251846b55de8792e5b5ef97264cfc.pdf).

