import numpy as np
from scipy.stats import norm


# SC-WEAT function
# ource: https://github.com/wolferobert3/gender_bias_swe_aies2022/blob/main/gender_association_collector.py
def SC_WEAT(w, A, B, permutations):
    w_normed = w / np.linalg.norm(w)
    A_normed = A / np.linalg.norm(A, axis=-1, keepdims=True)
    B_normed = B / np.linalg.norm(B, axis=-1, keepdims=True)

    A_associations = w_normed @ A_normed.T
    B_associations = w_normed @ B_normed.T
    joint_associations = np.concatenate((A_associations, B_associations), axis=-1)

    test_statistic = np.mean(A_associations) - np.mean(B_associations)
    effect_size = test_statistic / np.std(joint_associations, ddof=1)

    midpoint = len(A)
    sample_distribution = np.array([np.random.permutation(joint_associations) for _ in range(permutations)])
    sample_associations = np.mean(sample_distribution[:, :midpoint], axis=1) - np.mean(
        sample_distribution[:, midpoint:], axis=1)
    p_value = 1 - norm.cdf(test_statistic, np.mean(sample_associations), np.std(sample_associations, ddof=1))

    return effect_size, p_value


def get_word_vecs(model, words, print_every=1000):
    embeds = np.expand_dims(model[words[0]], axis=0)
    for i, word in enumerate(words[1:]):
        if i % print_every == 0:
            print(f'working on word {i}')
        embeds = np.append(embeds, np.expand_dims(model[word], axis=0), axis=0)
    return embeds
