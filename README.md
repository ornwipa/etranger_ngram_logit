# L'étranger: ngram feature selection and classification

## Introduction

The novel, **l’Étranger** (1942) by Albert Camus, has two parts that represent distictive sentiments. Part 1, before the crime incident, the main character's narration was indifferent and mostly about the "normal" day. Part 2, after the murder, there "may" appear more emotional statement during the time in prison and the trial. 

This program is an attempt to classify a given paragraph in which part it is. The goal is to demonstrate (1.) a simple feature extraction method in natural language processing and (2.) a use of supervised learning technique.

## Methods

### N-gram

Text data were initially splitted into paragraphs as labels for the part of the book were added as shown in [*data_prep_example*](https://github.com/ornwipa/etranger_ngram_logit/blob/master/data_prep_example). This dataframe served as a corpus for feature extraction.

The unigram, bigram and trigram terms were extracted using TF-IDF-vectorizer.
- TF: Term Frequency, TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
- IDF: Inverse Document Frequency, IDF(t) = log_e(Total number of documents / Number of documents with term t in it)

### Logistic Regression

A subset of 75% data were used for training a logistic regression model. Additionally, 3-fold cross-validation was conducted during the training. The rest 25% of the data were used for testing the model and computing model accuracy.

Note that this classification is not for the purpose of making an inference or detecting whether the text in the two parts are different with any "statistical significance".

## Results

The extracted "vocabulary" features is stored together with model output. The logit model output with coefficients of all features stored in [*model_coef.csv*](https://github.com/ornwipa/etranger_ngram_logit/blob/master/model_coef.csv).

In addition, the [*model_output_example*](https://github.com/ornwipa/etranger_ngram_logit/blob/master/model_output_example) shows the printout of the coefficents of the top-5 (-) features indicating that the paragraph belongs to part 1 and the top-5 (+) features indicating that the paragraph belongs to part 2.

## Interpretation

Considering the content of the book, the resulted dominant terms (top-5 positive and top-5 negative coefs) were as expected. Certainly, the terms "raymond", "concierge", "agent", "vieux" and "soleil" could predict the paragraph to be in part 1, and the terms "avocat", "procureur", "président" and "prison" mainly appeared in part 2. Interestingly, the term "homme" (coef = 8.4) would have been expected to reflect something very general, but in this case "homme" might be a reference to someone a speaker perceived as too insignificant to have the name mentioned thus rather occurred in part 2.

Furthermore, the terms with some sentiment were leaning towards classifying a paragraph to part 2; those terms are "cœur" (coef = 6.8), "visage" (coef = 4.8), "jamais" (coef = 6.9), "personne" (coef = 3.9).

The ngram terms, in addition to one word, are valuable. The following are examples of this importance:
- The word "faisait" (coef = "-0.4") alone could suggest for either parts and "mal" (coef = -1.3) alone would lean towards part 1; however, "faisait mal" (coef = 2.3) would indeed suggest that the paragraph was in part 2.
- The phrase "répondu non" (coef = -2.0) seemed to indicate an indifferent feeling towards someone else's propositions in part 1 whereas separated terms "répondre" (coef = 4.3), "répondu" (coef = 2.5) and "non" (coef = 3.7) would indicate the text might be in part 2 which covers more interogations.
- The word "temps" (coef = 1.0) did not appear to give much information. However, when bigram is included, the term "temps temps" (coef = -2.7), which probably came from "de temps en temps" after stop words were applied, could provide suggestion that a paragraph may be classified in part 1.

Similar to the [word clouds](https://github.com/ornwipa/etranger_word_cloud), the terms that include "tout" and "bien" suggested for part 2. Those terms are "comme tout monde", "tout monde", "tout", "toute", "toute façon", "bien", "connaissait bien", "si bien", "très bien". The term "plus" also had positive coefficients but "plus" with other terms such as "plus loin" and "rien plus" yielded diverse results.

### Limitation

Word stemming and lemmatization were attempted but not used because they ended up causing confusion for some words. Still, the prediction model already performed well with existing features.
