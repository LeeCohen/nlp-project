from PCFG import PCFG
from cky import load_sents_to_parse, cky

if __name__ == '__main__':
    pcfg = PCFG.from_file_assert_cnf('grammar3-CNF.txt')
    good_sents = load_sents_to_parse('sents_3.txt')
    bad_sents = load_sents_to_parse('sents_bad.txt')
    print '========'
    print 'Checking good sentences!'
    print '========'
    failures = 0
    for sent in good_sents:
        if "FAILED" in cky(pcfg, sent):
            print 'FAILURE!!! Failed to parse %s' % sent
            failures += 1
    print 'Succeeded %d out of %d' % (len(good_sents) - failures, len(good_sents))

    print '========'
    print 'Checking bad sentences!'
    print '========'
    failures = 0
    for sent in bad_sents:
        if "FAILED" not in cky(pcfg, sent):
            print 'FAILURE!!! Parsed when it should have failed: %s' % sent
    print 'Succeeded %d out of %d' % (len(bad_sents) - failures, len(bad_sents))
