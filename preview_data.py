import seaborn as sb
import matplotlib.pyplot as plt


def preview(data):
    types_counts = data['type'].value_counts()

    plt.figure(figsize=(12, 4))
    plt.ylabel('Counts')
    plt.xlabel('Types', fontsize=10)
    types_counts.plot.bar()
    plt.show()