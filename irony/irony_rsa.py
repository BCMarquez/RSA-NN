#!/usr/bin/python3
import math
import json


states = [ "terrible", "bad", "neutral", "good", "amazing"]
utterances = [ "terrible", "bad", "neutral", "good", "amazing"]
quds = [ "state", "valence", "arousal"]
valences = ["positive", "negative"]
arousals = ["high", "low"]
affects = [(v, a) for v in valences for a in arousals]
contexts = ["WC1", "WC2","WC3","WC4","WC5","WC6","WC7","WC8","WC9"]

# For each context, prior over the states
prior_states = eval(open('irony/prior_states.json', 'r').read())
print("prior states: ", prior_states)
# For each state, prior over valence+arousal combination
prior_affect = eval(open('irony/prior_affect.json', 'r').read())

# Prior over QUDs
# hmmm was this ever calculated in the paper?
prior_quds = {
    "state":    0.3,
    "valence":  0.3,
    "arousal":  0.4,
}

rationality_factor = 1.0

# the q function in the paper
# 's' is the state of the world, 'A' reps the speaker's affect toward the state
# this serves as a projection from full meaning spact to subset of speaker's interest
def qud(q, s, A):
    if q == "state": return s
    if q == "valence": return A[0]
    if q == "arousal": return A[1]
    print("error")


def literal_listener(s, A, u):
    if s != u:          #if state does not == utterance. This is the basic RSA that returns priors for true states.
        return 0.0
    else:
        return prior_affect[s][A] #so the literal listener will return literally whatever the qud asks for. but what happens if the qud is the state? should we return the probability of a weather state happening? If it doesn't it probs won't change behavior because they're all uniform.


# the U function in the paper, without the log
# this models info gained by listener about topic of interest
# where q is the intended QUD

def exp_utility(u, s, A, q):
    sum = 0.0
    for sp in states:
        for Ap in affects:
            if qud(q, s, A) == qud(q, sp, Ap):
                sum += literal_listener(sp, Ap, u)
    return sum

# the S function in the paper, normalized
# "the speaker S chooses an utterance according to a softmax decision rule"
# in the paper, it says the base is e. How come it's the the utility ftn^ rationality factior here?
def speaker(u, s, A, q):
    norm = 0.0
    for up in utterances:
        norm += math.pow(exp_utility(up, s, A, q), rationality_factor)
    return math.pow(exp_utility(u, s, A, q), rationality_factor) / norm

# the pragmatic L function in the paper, unnormalized
# this is the ftn that models the ambiguity behind QUD, particularly it's held in the sum
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
