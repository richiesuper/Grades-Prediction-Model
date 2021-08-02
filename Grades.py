import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

original_df = pd.read_csv('test_scores.csv')


def test_scores:
    test_results = pd.DataFrame({'Pretest': original_df['pretest'], 'Posttest': original_df['posttest']})
    test_results.plot()
    plt.show()
