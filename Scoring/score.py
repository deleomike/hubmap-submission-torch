import pandas as pd


def score(submissionFilePath, testImagesPath):

    df = pd.read_csv(submissionFilePath)

    for item in df:
        # Interpret pixel list from csv

        # Import pixel list from json
        pass


    print(df)

    return 0

if __name__ == "__main__":
    sample = "./input/hubmap-kidney-segmentation/sample_submission.csv"
    prediction = "./submission.csv"
    res = score(sample, "./input/hubmap-kidney-segmentation/test/")

    print("Score of {}".format(res))