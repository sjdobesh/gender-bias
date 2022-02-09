#!/usr/bin/env python3
'''
@author : samantha dobesh
@date : Feb 7th 2022
@desc : Very basic file showing a simplified version of the masking
        and evaluation steps for BERT's gender bias
'''


import re
import math
import pandas as pd
from tqdm import tqdm
from transformers import pipeline


# debug toggle
DBUG = False
# load unmasking pipeline
bert = pipeline('fill-mask', model='bert-base-uncased')


def debug(string, obj=None):
    '''
    debug print that can be disabled w constant
    '''
    if DBUG:
        print("[DEBUG] ")
        print(string)
        if obj is not None:
            print("\n")
            print(obj)


def toy_data():
    '''
    quickly load some toy data into namespace for prompt
    '''
    return (
        ['he is a computer scientist.', 'she is a computer scientist.'],
        ['he', 'she'],
        ['computer', 'scientist']
    )


def unmask_sentences(sentences):
    '''
    unmask a set of sentences and return them as a list
    '''
    unmasked_sentences = []
    for sentence in tqdm(sentences):
        results = bert(sentence)
        debug("predictions--")
        debug("original : " + sentence)
        for result in results:
            debug("prediction : " + result['sequence'])
            unmasked_sentences.append(result['sequence'])
    return unmasked_sentences


def test_single():
    '''test a single measurement'''

    print("\nSingle Evaluation...\n")

    sentence = 'he is a scientist.'
    target_mask = '[MASK] is a scientist.'
    prior_mask = '[MASK] is a [MASK].'

    print("\n**** step 1: ****")
    print("take a sentence with target and attributes")
    print('\'' + sentence + '\'')

    print("\n**** step 2: ****")
    print("mask targets")
    print('\'' + target_mask + '\'')

    print("\n**** step 3: ****")
    print("obtain probability of masked token")
    he_probability = bert(target_mask, targets='he')[0]['score']
    print("he  : ", he_probability)
    she_probability = bert(target_mask, targets='she')[0]['score']
    print("she : ", she_probability)

    print("\n*** step 4: ****")
    print("mask out attributes ****")
    print('\'' + prior_mask + '\'')

    print("\n**** step 5: ****")
    print("obtain probabilities without attributes")
    he_prior_probability = bert(prior_mask, targets='he')
    print("he  : ", he_prior_probability[0][0]['score'])
    she_prior_probability = bert(prior_mask, targets='she')
    print("she : ", she_prior_probability[0][0]['score'])

    print("\n**** step 6: ****")
    print("log score:")
    he_logscores = math.log(
        he_probability / he_prior_probability[0][0]['score']
    )
    she_logscores = math.log(
        she_probability / she_prior_probability[0][0]['score']
    )
    print("he  : ", he_logscores)
    print("she : ", she_logscores)


def mask_words(words, sentences):
    '''
    mask a list of words for a list of sentences
    returns a list of tuples (masked word, masked sentence)
    '''
    # if given a single word, wrap it in a list
    if isinstance(words, str):
        words = [words]
    # generate masked list
    masked = []
    for sentence in sentences:
        found = []
        for word in words:
            regex = r'\b' + word + r'\b'
            # check if the word exists
            if len(re.findall(regex, sentence)) > 0:
                # if so, swap for [MASK]
                sentence = re.sub(regex, '[MASK]', sentence)
                found.append(word)
        masked.append((found, sentence))
    return masked


def generate_data(sentences, targets, attributes):
    '''
    generate data structure suitable for batch tests.
    each entry in list is a tuple of
    (target words, target mask, attribute words, attribute mask)
    '''
    target_masks = mask_words(targets, sentences)
    target_hits = []
    target_sentences = []
    for mask in target_masks:
        target_hits.append(mask[0])
        target_sentences.append(mask[1])
    prior_hits = []
    prior_sentences = []
    prior_masks = mask_words(attributes, target_sentences)
    for mask in prior_masks:
        prior_hits.append(mask[0])
        prior_sentences.append(mask[1])
    data = []
    for i in range(len(target_sentences)):
        data.append((
            target_hits[i],
            target_sentences[i],
            prior_hits[i],
            prior_sentences[i]
        ))
    return data


def test_data(data):
    '''
    test generated data for biases
    returns a result tuple
    (token, pT, attributes, pP, score)
    '''
    results = []
    for datum in tqdm(data):
        token_p = (bert(datum[1], targets=datum[0])[0]['score'])
        prior_p = (bert(datum[3], targets=datum[0])[0][0]['score'])
        score = (math.log(token_p/prior_p))
        results.append((datum[0], token_p, datum[2], prior_p, score))
    return results


def print_results(results):
    '''print out all the results'''
    for result in results:
        print("token:", result[0])
        print("token probability:", result[1])
        print("attributes:", result[2])
        print("prior probabilities:", result[3])
        print("log score:", result[4])
        print("")


def main():
    '''main, generate data and do junk with it'''
    sentences, targets, attributes = toy_data()
    data = generate_data(sentences, targets, attributes)
    results = test_data(data)
    print_results(results)


if __name__ == "__main__":
    main()
