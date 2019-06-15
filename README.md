# NHL-Prospect-Classifier
## Usage
Run `python main.py` to create corpus, train the model and get predictions. Experiment with the threshold parameter to change the discriminant for classes. Default is mean of draft class. The list of available arguments are

```sh
optional arguments:
  -h, --help         show this help message and exit
  -t , --threshold   String or integer determining threshold of games played
                     to be considered positive class.
  -q, --quiet        Print quiet
  -v, --verbose      Print verbose
  -l, --long         Model on long reports only
  -a, --all          Model on all reports
```
## Data
The scraped (and merged) data can be found in [data/merged/](https://github.com/mattjliu/NHL-Prospect-Classifier/tree/master/data/merged). The predictions for the current long report model can be found in [predictions/long/](https://github.com/mattjliu/NHL-Prospect-Classifier/tree/master/predictions/long).
