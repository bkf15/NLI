import string

# Char Vocabulary
# ------------------------------------------------------
# In training data...
# v had 56068 appearances.
# R had 590 appearances.
# ' had 11988 appearances.
# _ had 18 appearances.
# ( had 528 appearances.
# n had 363801 appearances.
# t had 525302 appearances.
# s had 318201 appearances.
# T had 12078 appearances.
# x had 4829 appearances.
# \ had 53 appearances.
# B had 2594 appearances.
# h had 278650 appearances.
# q had 1274 appearances.
# # had 2 appearances.
# W had 2644 appearances.
# 4 had 123 appearances.
# $ had 18 appearances.
# D had 619 appearances.
# ` had 3038 appearances.
# " had 0 appearances.
# ^ had 7 appearances.
# ~ had 6 appearances.
# i had 328729 appearances.
# Q had 11 appearances.
# 2 had 477 appearances.
# o had 416612 appearances.
# c had 113091 appearances.
# A had 5000 appearances.
# O had 2343 appearances.
# + had 26 appearances.
# % had 81 appearances.
# - had 1545 appearances.
# d had 168879 appearances.
# * had 23 appearances.
# V had 483 appearances.
# 0 had 858 appearances.
# z had 3563 appearances.
# ) had 668 appearances.
# p had 68403 appearances.
# } had 5 appearances.
# Y had 1427 appearances.
# 3 had 189 appearances.
# k had 40806 appearances.
# U had 497 appearances.
# a had 366206 appearances.
# H had 1524 appearances.
# u had 134584 appearances.
#   had 0 appearances.
# r had 257937 appearances.
# ! had 832 appearances.
# ] had 15 appearances.
# f had 101459 appearances.
# N had 1370 appearances.
# 7 had 77 appearances.
# { had 3 appearances.
# [ had 15 appearances.
# E had 1805 appearances.
# w had 101348 appearances.
# 8 had 102 appearances.
# L had 942 appearances.
# g had 104240 appearances.
# C had 860 appearances.
# e had 639677 appearances.
# J had 348 appearances.
# M had 1432 appearances.
# & had 66 appearances.
# Z had 17 appearances.
# I had 22270 appearances.
# @ had 0 appearances.
# F had 2839 appearances.
# G had 451 appearances.
# 5 had 219 appearances.
# / had 353 appearances.
# : had 874 appearances.
# ; had 682 appearances.
# ? had 1421 appearances.
# 6 had 113 appearances.
# P had 441 appearances.
# . had 902 appearances.
# | had 29 appearances.
# j had 8634 appearances.
# m had 112581 appearances.
# 1 had 430 appearances.
# , had 52544 appearances.
# y had 140665 appearances.
# K had 247 appearances.
# = had 25 appearances.
# X had 40 appearances.
# b had 57567 appearances.
# l had 211057 appearances.
# < had 8 appearances.
# 9 had 140 appearances.
# > had 25 appearances.
# S had 3568 appearances.
# Unknown had 1 appearances.

_vocabulary = set((list(string.ascii_letters) 
        + list(string.digits)
        + list(string.punctuation)))

vocabulary = {c : i + 1 for i, c in enumerate(_vocabulary)}
vocab = lambda c: vocabulary[c] if c in vocabulary else len(vocabulary) + 1
vocab_len = len(vocabulary) + 2

# Classes
# --------------------------------------------------------------
_classes = ['TUR', 'ARA', 'HIN', 'KOR', 'ITA', 'ZHO', 'TEL', 'SPA', 'JPN', 'FRA', 'DEU']
classes = {cl : i for i, cl in enumerate(_classes)}

# File paths
# --------------------------------------------------------------
tokenized_train = 'TOEFL11-TRAIN/data/text/responses/tokenized/'
train_labels = 'TOEFL11-TRAIN/data/text/index-training.csv'
tokenized_dev = 'TOEFL11-DEV/data/text/responses/tokenized/'
dev_labels = 'TOEFL11-DEV/data/text/index-dev.csv'
google_pretrained_word2vec = 'GoogleNews-vectors-negative300.bin'


# Upper Bounds
# -------------------------------------------------------------
# In training data ...
# Document Length 97 Percentile:
#         32.0
# Sentence Length 97 Percentile:
#         45.0
# Word Length 97 Percentile:
#         10.0

doc_len_ub = 32
sent_len_ub = 45
word_len_ub = 10
word_vec_size = 300