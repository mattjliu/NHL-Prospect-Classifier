from corpus import create_corpus
from model import train_model


def main(threshold=20, long_reports=True):
    create_corpus(threshold=threshold)
    train_model(long_reports=long_reports)


if __name__ == '__main__':
    main()
