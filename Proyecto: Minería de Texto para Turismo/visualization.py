import matplotlib.pyplot as plt
from tsne import tsne
import numpy as np


# %%
def make_visual(vec_rprstn, t_words, subsetwords, stop_words):
    # Proyección a dos dimensiones

    reduced_matrix = tsne(vec_rprstn, 2)

    # Visualizar constelación

    max_x = np.amax(reduced_matrix, axis=0)[0]
    max_y = np.amax(reduced_matrix, axis=0)[1]

    plt.figure(figsize=(40, 40), dpi=100)
    plt.xlim((-max_x, max_x))
    plt.ylim((-max_y, max_y))
    plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], 20, color='black')

    for idx, word in enumerate(t_words[:]):
        x = reduced_matrix[idx, 0]
        y = reduced_matrix[idx, 1]
        if word in stop_words:
            plt.annotate(word, (x, y), color='red')  # Coloreamos una categorías, la de stopwords
        else:
            plt.annotate(word, (x, y), color='black')

    plt.show()

    # Visualizar los vectores de algunas palabras

    plotted_subsetword = []
    subreduced_matrix = []

    for idx, word in enumerate(t_words[:]):
        if word in subsetwords:
            plotted_subsetword += [word]
            subreduced_matrix += [reduced_matrix[idx]]

    subreduced_matrix = np.array(subreduced_matrix)
    fig, ax = plt.subplots(figsize=(15, 15))
    col1 = 0
    col2 = 1

    # Dibujamos una flechita por palabra

    for word in subreduced_matrix:
        ax.arrow(0, 0, word[col1], word[col2], head_width=0.8, head_length=0.8, fc="r", ec="r", width=1e-2)

    ax.scatter(subreduced_matrix[:, col1], subreduced_matrix[:, col2])  # Punto por palabra del subconjunto

    # Nombre de cada palabra

    for i in range(0, len(plotted_subsetword)):
        ax.annotate(plotted_subsetword[i], (subreduced_matrix[i, col1], subreduced_matrix[i, col2]))

    plt.show()
