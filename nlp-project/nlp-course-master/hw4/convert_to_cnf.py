import sys
import os

import itertools

import PCFG


PASTE_START = "# PASTE START"
PASTE_END = "# PASTE END"

def paste_in_file(dst, content):
    with open(dst, "rb") as original_dst:
        dst_file_content = original_dst.read()
    assert dst_file_content.count(PASTE_START) == 1
    assert dst_file_content.count(PASTE_END) == 1
    assert PASTE_START not in content and PASTE_END not in content
    s = dst_file_content.find(PASTE_START) + len(PASTE_START)
    e = dst_file_content.find(PASTE_END)
    with open(dst, "wb") as dst_out:
        dst_out.write(dst_file_content[:s])
        dst_out.write("\n")
        dst_out.write(content)
        dst_out.write("\n")
        dst_out.write(dst_file_content[e:])


def replace_element(x, src, dst):
    if x == src:
        return dst
    else:
        return x


def list_replace_options(l, src, dst):
    indices = [i for i in xrange(len(l)) if l[i] == src]
    for r in xrange(1, len(indices) + 1):
        for subset in itertools.combinations(indices, r):
            val = list(l)
            for i in subset:
                val[i] = dst
            yield val


def add_unary_rule(pcfg, src, dst):
    print "replacing ", src, dst
    for symbol, weighted_derivations in pcfg._rules.items():
        for symbols_list, weight in weighted_derivations:
            if src in symbols_list:
                for l in list_replace_options(symbols_list, src, dst):
                    pcfg.add_rule(symbol, l, weight)
    return pcfg


def remove_unary_rules(pcfg):
    res = PCFG.PCFG()
    rules = pcfg._rules
    src = None
    dsts = []
    for symbol, weighted_derivations in rules.iteritems():
        for symbols_list, weight in weighted_derivations:
            if src in (None, symbol) and len(symbols_list) == 1 and not symbols_list[0].islower():
                src = symbol
                dsts.append(symbols_list[0])
            else:
                res.add_rule(symbol, symbols_list, weight)
    if src is not None:
        for dst in dsts:
            add_unary_rule(res, src, dst)
    if res.is_terminal(src):
        res = remove_rules_with_symbol(res, src)
    return res, src is not None


def remove_rules_with_symbol(pcfg, symbol_to_ignore):
    res = PCFG.PCFG()
    for symbol, weighted_derivations in pcfg._rules.iteritems():
        for symbols_list, weight in weighted_derivations:
            if symbol_to_ignore not in symbols_list:
                res.add_rule(symbol, symbols_list, weight)
    return res


def add_symbol(symbol, list_symbols):
    for i in xrange(1, 500):
        suggestion = "{}_{}".format(symbol, i)
        if suggestion not in list_symbols:
            list_symbols.append(suggestion)
            return suggestion
    print symbol
    assert False


def reduce_to_binary_rules(pcfg):
    res = PCFG.PCFG()
    all_symbols = pcfg._rules.keys()
    for symbol, weighted_derivations in pcfg._rules.iteritems():
        for symbols_list, weight in weighted_derivations:
            if len(symbols_list) > 2:
                current_symbol = symbol
                for s in symbols_list[:-2]:
                    new_symbol = add_symbol(symbol, all_symbols)
                    res.add_rule(current_symbol, [s, new_symbol], weight)
                    current_symbol = new_symbol
                res.add_rule(current_symbol, symbols_list[-2:], 1)
            else:
                res.add_rule(symbol, symbols_list, weight)
    return res


def create_cnf_rules(pcfg):
    for i in xrange(30):
        pcfg, changed = remove_unary_rules(pcfg)
        if not changed:
            break
    return reduce_to_binary_rules(pcfg)


def to_strings(pcfg):
    res = []
    for symbol, weighted_derivations in pcfg._rules.iteritems():
        for symbols_list, weight in weighted_derivations:
            res.append("{}\t{}\t{}".format(weight, symbol, " ".join(symbols_list)))
    return res


def generate_file(source, dst):
    assert os.path.exists(dst) and os.path.exists(source)
    cnf_pcfg = create_cnf_rules(PCFG.PCFG.from_file(source))
    content = "\n".join(to_strings(cnf_pcfg))
    paste_in_file(dst, content)
    print "DONE - validating..."
    PCFG.PCFG.from_file_assert_cnf(dst)


if __name__ == '__main__':
    generate_file(sys.argv[1], sys.argv[2])
