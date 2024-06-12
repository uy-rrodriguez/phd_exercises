"""
Tools to train and execute SVM model.
"""

from liblinear.liblinearutil import *


def main() -> None:
    """
    Main program.
    """
    # Read training data in LIBSVM format
    y, x = svm_read_problem("svm_in/train.svm")
    m = train(y, x, "-c 4")

    # Predict with test data
    p_y, p_x = svm_read_problem("svm_in/test.svm")
    p_label, p_acc, p_val = predict(p_y, p_x, m)


if __name__ == '__main__':
    main()
