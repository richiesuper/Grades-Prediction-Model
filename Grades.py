import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
df = pd.read_csv('test_scores.csv')


def show_correlation():
    test_results = pd.DataFrame({'Pretest': df['pretest'], 'Posttest': df['posttest']})
    test_results.plot()
    plt.show()

show_correlation()