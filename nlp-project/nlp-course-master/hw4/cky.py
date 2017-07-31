import sys
from collections import defaultdict

from PCFG import PCFG


def load_sents_to_parse(filename):
    sents = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                sents.append(line)
    return sents


def cky(pcfg, sent):
    """
    Calculates the most likely derivation.

    :param pcfg: a CNF PCFG.
    :param sent:
    :return: derivation tree.
    """
    ### YOUR CODE HERE
    split_sent = sent.split()
    n = len(split_sent)
    pi = defaultdict(float)
    bp = {}
    rules = pcfg._rules
    for i in range(1, n + 1):
        set_flag = False
        curr_word = split_sent[i - 1]
        for X, derivations in rules.iteritems():
            for derivation in derivations:
                if [curr_word] == derivation[0]:
                    pi[(i, i, X)] = derivation[1]
                    bp[(i, i, X)] = Node(X, None, None, curr_word)
                    set_flag = True
        if not set_flag:
            return "FAILED TO PARSE!"
    for i in reversed(range(1, n)):
        for l in xrange(1, n - i + 1):
            j = i + l
            for X, derivations in rules.iteritems():
                max_value = 0
                for derivation in derivations:
                    if len(derivation[0]) == 2:
                        for s in xrange(i, j):
                            key_left = (i, s, derivation[0][0])
                            key_right = (s + 1, j, derivation[0][1])
                            value = derivation[1] * pi[key_left] * pi[key_right]
                            max_value = max(value, max_value)
                            if value == max_value and value > 0:
                                bp[(i, j, X)] = Node(X, bp[key_left], bp[key_right], None)
                                pi[(i, j, X)] = max_value

    whole_key = (1, n, 'ROOT')
    if whole_key in bp:
        node = bp[whole_key]
        return get_tree(node)
    ### END YOUR CODE
    return "FAILED TO PARSE!"


def get_tree(root):
    """
    getParseTree() takes a root and constructs the tree in the form of a
    string. Right and left subtrees are indented equally, providing for
    a nice display.
    @params: root node and an indent factor (int).
    @return: tree, starting at the root provided, in the form of a string.
    """
    if root.terminal and root.root:
        return '(' + root.root + ' ' + root.terminal + ')'
    if root.terminal:
        return root.terminal

    # Calculates the new indent factors that we need to pass forward.
    left = get_tree(root.left)
    right = get_tree(root.right)
    return '(' + root.root + ' ' + left + ' ' + right + ')'


class Node(object):
    def __init__(self, root, left, right, terminal):
        self.root = root
        self.left = left
        self.right = right
        self.terminal = terminal


if __name__ == '__main__':
    pcfg = PCFG.from_file_assert_cnf(sys.argv[1])
    sents_to_parse = load_sents_to_parse(sys.argv[2])
    for sent in sents_to_parse:
        print cky(pcfg, sent)
