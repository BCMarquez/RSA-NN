#!/usr/bin/python3
import math

states = [ "terrible", "bad", "neutral", "good", "amazing"]
utterances = [ "terrible", "bad", "neutral", "good", "amazing"]
quds = [ "state", "valence", "arousal"]
valences = ["positive", "negative"]
arousals = ["high", "low"]
affects = [(v, a) for v in valences for a in arousals]
contexts = ["WC1", "WC6", "WC9"]

# For each context, prior over the states
prior_states = {
    "WC1" : {
        "terrible": 0.015,
        "bad":      0.02,
        "neutral":  0.03,
        "good":     0.035,
        "amazing":  0.9
    },
    "WC6" : {
        "terrible": 0.05,
        "bad":      0.14,
        "neutral":  0.5,
        "good":     0.26,
        "amazing":  0.05
    },
    "WC9" : {
        "terrible": 0.55,
        "bad":      0.3,
        "neutral":  0.07,
        "good":     0.05,
        "amazing":  0.03
    }
}
# For each state, prior over valence+arousal combination
prior_affect = {
    "terrible": {
        ("positive", "high"): 0.05,
        ("negative", "high"): 0.7,
        ("positive", "low"):  0.05,
        ("negative", "low"):  0.2,
    },
    "bad": {
        ("positive", "high"): 0.05,
        ("negative", "high"): 0.5,
        ("positive", "low"):  0.05,
        ("negative", "low"):  0.4,
    },
    "neutral": {
        ("positive", "high"): 0.05,
        ("negative", "high"): 0.05,
        ("positive", "low"):  0.45,
        ("negative", "low"):  0.45,
    },
    "good": {
        ("positive", "high"): 0.4,
        ("negative", "high"): 0.05,
        ("positive", "low"):  0.5,
        ("negative", "low"):  0.05,
    },
    "amazing": {
        ("positive", "high"): 0.7,
        ("negative", "high"): 0.05,
        ("positive", "low"):  0.2,
        ("negative", "low"):  0.05,
    },
}
# Prior over QUDs
prior_quds = {
    "state":    0.3,
    "valence":  0.3,
    "arousal":  0.4,
}

rationality_factor = 1.0

# the q function in the paper
def qud(q, s, A):
    if q == "state": return s
    if q == "valence": return A[0]
    if q == "arousal": return A[1]
    print("error")

def literal_listener(s, A, u):
    if s != u: 
        return 0.0
    else:
        return prior_affect[s][A]

# the U function in the paper, without the log
def exp_utility(u, s, A, q):
    sum = 0.0
    for sp in states:
        for Ap in affects:
            if qud(q, s, A) == qud(q, sp, Ap):
                sum += literal_listener(sp, Ap, u)
    return sum

# the S function in the paper, normalized
def speaker(u, s, A, q):
    norm = 0.0
    for up in utterances:
        norm += math.pow(exp_utility(up, s, A, q), rationality_factor)
    return math.pow(exp_utility(u, s, A, q), rationality_factor) / norm

# the pragmatic L function in the paper, unnormalized
def unnorm_pragmatic_listener(s, A, u, context):
    sum = 0.0
    for q in quds:
        sum += prior_quds[q] * speaker(u, s, A, q)
    return prior_states[context][s] * prior_affect[s][A] * sum

# the pragmatic L function in the paper, normalized
def pragmatic_listener(s, A, u, context):
    norm = 0.0
    for sp in states:
        for Ap in affects:
            norm += unnorm_pragmatic_listener(sp, Ap, u, context)
    return unnorm_pragmatic_listener(s, A, u, context) / norm


def main():
    for context in contexts:
        print("----------------------")
        print("  CONTEXT: %s" % context)
        print("----------------------")
        for u in utterances:
            print("--- utterance: %s ----" % u)
            total_prob = 0.0
            for s in states:
                for A in affects:
                    prob = pragmatic_listener(s, A, u, context)
                    print("  s: %s, A: %s:\t%f" % (s, A, prob))
                    total_prob += prob
            print("  total_prob: %f" % total_prob)

if __name__ == "__main__": main()