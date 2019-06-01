from corpus import create_corpus
from model import train_and_predict
import argparse


def main(threshold, long_reports, verbose):
    create_corpus(threshold=threshold, verbose=verbose)
    train_and_predict(long_reports=long_reports, verbose=verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define target of model')
    parser.add_argument('-t', '--threshold', dest='threshold', metavar='', type=str,
                        help='Function or integer determining threshold of games played to be considered positive class.')
    print_group = parser.add_mutually_exclusive_group()
    print_group.add_argument('-q', '--quiet', action='store_const', dest='verbose', const=False, help='Print quiet')
    print_group.add_argument('-v', '--verbose', action='store_const', dest='verbose', const=True, help='Print verbose')
    report_group = parser.add_mutually_exclusive_group()
    report_group.add_argument('-l', '--long', action='store_const', dest='long', const=True, help='Model on long reports only')
    report_group.add_argument('-a', '--all', action='store_const', dest='long', const=False, help='Model on all reports')
    parser.set_defaults(threshold='mean', verbose=True, long=True)
    args = parser.parse_args()

    if args.threshold.isdigit():
        args.threshold = int(args.threshold)
    main(threshold=args.threshold, long_reports=args.long, verbose=args.verbose)
