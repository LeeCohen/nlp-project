import os
import utils
import sys
import time
import cPickle
import collections
import random
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from data import *

BEGIN_TAG = '*'
END_TAG = 'STOP'


def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Rerutns: The word's features.
    """
    features = {}
    ### YOUR CODE HERE
    # All words
    features['word'] = curr_word
    features['next_word'] = next_word
    features['prev_word'] = prev_word
    features['prevprev_word'] = prevprev_word

    # All tags
    features['prev_tag'] = prev_tag
    features['prevprev_tag'] = prevprev_tag
    features['tag_bigram'] = prev_tag + ' ' + prevprev_tag

    # Capitalized
    features['is_cap'] = curr_word[0].isupper() and prev_tag != BEGIN_TAG
    features['prev_is_cap'] = prev_word[0].isupper() and prevprev_tag != BEGIN_TAG
    features['next_is_cap'] = next_word[0].isupper()


    # To reduce features amount, we did not use the folowing features:
    # features['prev_word_tag'] = prev_word + ' ' + prev_tag
    # features['prevprev_word_tag'] = prevprev_word + ' ' + prevprev_tag
    if not features['is_cap']:
        curr_word = curr_word.lower()
        for suffix in SUFFIXES:
            features['suffix_{}'.format(suffix)] = int(curr_word.endswith(suffix))

        for prefix in PREFIXES:
            features['prefix_{}'.format(prefix)] = int(curr_word.startswith(prefix))

    ### END YOUR CODE
    return features

def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<s>', BEGIN_TAG)
    prevprev_token = sentence[i - 2] if i > 1 else ('<s>', BEGIN_TAG)
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', END_TAG)
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])


def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Rerutns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)


def create_examples(sents):
    print "building examples"
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in xrange(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tagset[sent[i][1]])
    return examples, labels


def memm_greeedy(sent, logreg, vec):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    sent_with_tags = map(lambda i: [sent[i], ''], range(len(sent)))
    for i in xrange(len(sent)):
        features = extract_features(sent_with_tags, i)
        vec_features = vectorize_features(vec, features)
        predicted_tags[i] = index_to_tag_dict[logreg.predict(vec_features).item()]
        sent_with_tags[i][1] = predicted_tags[i]

    ### END YOUR CODE
    return predicted_tags


t1_timer = t = utils.Timer("None")
PRUNE_THRESHOLD = 30


def prune_layer(key2value):
    if len(key2value) <= PRUNE_THRESHOLD:
        return key2value
    threshold = sorted(key2value.values())[-PRUNE_THRESHOLD]
    return {k: v for k, v in key2value.iteritems() if v >= threshold}


def memm_viterbi(sent, logreg, vec):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    global t1_timer
    ### YOUR CODE HERE
    layer = {(BEGIN_TAG, BEGIN_TAG): 0}
    back_pointers = []
    SMALL = -99999.
    # sizes = []

    for i, word in enumerate(sent):
        new_layer = collections.defaultdict(lambda: SMALL)
        current_back_pointers = {}
        t1_timer.start_part("prediction")
        wu2probs = array_calculate_tagging_probs(sent, i, layer.keys(), logreg, vec)

        t1_timer.start_part("loop")
        for tag_index, v in index_to_tag_dict.iteritems():
            for (w, u), prev_log_prob in layer.iteritems():
                current_log_probs = wu2probs[(w, u)]
                if current_log_probs[tag_index] < -10.:
                    continue
                log_prob_wuv = prev_log_prob + current_log_probs[tag_index]
                if log_prob_wuv > new_layer[(u, v)]:
                    new_layer[(u, v)] = log_prob_wuv
                    current_back_pointers[(u, v)] = w
        t1_timer.start_part("None")
        back_pointers.append(current_back_pointers)
        layer = {key: value for key, value in new_layer.iteritems() if value > SMALL}
        layer = prune_layer(layer)
        # sizes.append(len(layer))
    two_final_tags = max(layer.items(), key=lambda ((u, v), log_prob): log_prob)[0]
    predicted_tags = list(two_final_tags)
    for current_back_pointers in back_pointers[::-1][:-2]:
        w = current_back_pointers[tuple(predicted_tags[:2])]
        predicted_tags.insert(0, w)
    ### END YOUR CODE
    return predicted_tags


def array_calculate_tagging_probs(sent, i, w_u_list, logreg, vec):
    # w_u_list is list of (w, u)
    t1_timer.start_part("preprocess_prediction")
    all_features_vecs = []
    for w, u in w_u_list:
        sent_with_tags = map(list, zip(sent, [None] * len(sent)))
        if i > 1:
            sent_with_tags[i - 2][1] = w
        if i > 0:
            sent_with_tags[i - 1][1] = u
        features = extract_features(sent_with_tags, i)
        all_features_vecs.append(features)
    sparse_features = vec.transform(all_features_vecs)
    t1_timer.start_part("prediction")
    probs_vecs = np.log(logreg.predict_proba(sparse_features))
    return {(w, u): probs_vecs[j] for j, (w, u) in enumerate(w_u_list)}


def sample_bad_classifications_viterbi(original_test_data, test_data, logreg, vec):
    for i in random.sample(range(len(test_data)), 1000):
        original_sentence, sentence = original_test_data[i], test_data[i]
        words = [word for word, tag in sentence]
        original_words = [word for word, tag in original_sentence]
        viterbi_tags = memm_viterbi(words, logreg, vec)
        real_tags = [tag for (word, tag) in sentence]
        if equal_amount(real_tags, viterbi_tags) != len(real_tags):
            s = ''
            for original_word, word, tag, real_tag in zip(original_words, words, viterbi_tags, real_tags):
                if word != original_word:
                    s += '{} [replace={}] '.format(original_word, word)
                else:
                    s += word + ' '
                if real_tag != tag:
                    s += '(real={}, memm={}) '.format(real_tag, tag)
            print s


def memm_eval(test_data, logreg, vec):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm & greedy hmm
    """
    global t1_timer
    ### YOUR CODE HERE
    count_good_greedy = 0
    count_good_viterbi = 0
    count_total = 0
    start_time = time.time()
    N = 100
    for i_loop, sentence in enumerate(test_data):
        if i_loop % N == 1:
            elapsed = time.time() - start_time
            per_iteration_time = elapsed / i_loop
            print t1_timer
            print 'time left', (len(test_data) - i_loop) * per_iteration_time
            print 'acc greedy so far:', str(float(count_good_greedy) / count_total)
            print 'acc viterbi so far:', str(float(count_good_viterbi) / count_total)
        words = [word for word, tag in sentence]
        greedy_tags = memm_greeedy(words, logreg, vec)
        viterbi_tags = memm_viterbi(words, logreg, vec)
        real_tags = [tag for (word, tag) in sentence]
        count_good_greedy += equal_amount(real_tags, greedy_tags)
        count_good_viterbi += equal_amount(real_tags, viterbi_tags)
        count_total += len(real_tags)
    acc_greedy = str(float(count_good_greedy) / count_total)
    acc_viterbi = str(float(count_good_viterbi) / count_total)
    ### END YOUR CODE
    return acc_viterbi, acc_greedy


def equal_amount(l1, l2):
    return sum([1 for x, y in zip(l1, l2) if x == y])


def main():
    global index_to_tag_dict, tagset
    num_samples = int(sys.argv[1])
    FNAME = sys.argv[2]
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")[:num_samples]
    original_dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    random.shuffle(original_dev_sents)
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, original_dev_sents)

    #The log-linear model training.
    #NOTE: this part of the code is just a suggestion! You can change it as you wish!
    curr_tag_index = 0
    tagset = {}
    for train_sent in train_sents:
        for token in train_sent:
            tag = token[1]
            if tag not in tagset:
                tagset[tag] = curr_tag_index
                curr_tag_index += 1
    index_to_tag_dict = invert_dict(tagset)
    vec = DictVectorizer()
    print "Create train examples"
    train_examples, train_labels = create_examples(train_sents)
    num_train_examples = len(train_examples)
    print "#example: " + str(num_train_examples)
    print "Done"

    print "Create dev examples"
    dev_examples, dev_labels = create_examples(dev_sents)
    num_dev_examples = len(dev_examples)
    print "#example: " + str(num_dev_examples)
    print "Done"

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print "Vectorize examples"
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print "Done"
    print "fname", FNAME
    if os.path.exists(FNAME):
        with open(FNAME, "rb") as f_in:
            logreg = cPickle.load(f_in)
    else:
        logreg = linear_model.LogisticRegression(
            multi_class='multinomial', max_iter=128, solver='lbfgs', C=100000, verbose=1, n_jobs=8)
        print "Fitting..."
        start = time.time()
        logreg.fit(train_examples_vectorized, train_labels)
        with open(FNAME, "wb") as f_out:
            cPickle.dump(logreg, f_out)
        end = time.time()
        print "done, " + str(end - start) + " sec"
    #End of log linear model training
    sample_bad_classifications_viterbi(original_dev_sents, dev_sents, logreg, vec)

    acc_viterbi, acc_greedy = memm_eval(dev_sents, logreg, vec)
    print "dev: acc memm greedy: " + acc_greedy
    print "dev: acc memm viterbi: " + acc_viterbi
    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi, acc_greedy = memm_eval(test_sents, logreg, vec)
        print "test: acc memmm greedy: " + acc_greedy
        print "test: acc memmm viterbi: " + acc_viterbi


if __name__ == "__main__":
    main()
