"""Small hybrid verification fixture, not a pretrained model or serving engine."""

import torch


def causal_attention(query, keys, values):
    # The caller supplies only the causal prefix, so no future-token mask is needed.
    scores = (keys @ query) / (query.shape[0] ** 0.5)
    probabilities = torch.softmax(scores, dim=0)
    return probabilities @ values


def accepted_prefix(initial_logits, token_logits, proposed_tokens):
    # Logits AFTER token t predict token t+1. The first prediction comes from
    # the already processed context, not from looking at the first proposal.
    predictions = torch.cat((initial_logits.unsqueeze(0), token_logits[:-1]))
    matches = predictions.argmax(dim=-1) == proposed_tokens
    return matches.to(torch.int64).cumprod(dim=0).sum(dim=0, keepdim=True)


class TinyHybridWindow(torch.nn.Module):
    """Parallel recurrence/attention branches plus greedy prefix verification.

    Inputs: proposed_tokens [T] int64, initial_state [D, K], context_keys [P,K],
    context_values [P,D], initial_logits [vocab]. T >= 1. Context cache entries
    are all valid. Outputs include logits [T,vocab], accepted state [D,K],
    tentative KV buffers, valid KV prefix length [1], and accepted count [1].

    Rejected KV suffixes remain physically allocated but are logically invalid.
    A later consumer must honor valid_cache_length. Correction/bonus-token
    execution, sampling, batching, and multi-layer cache coordination are omitted.
    """

    def __init__(self, width=4, keys=3, vocab=7, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        def weight(*shape):
            return torch.randn(*shape, generator=generator, dtype=torch.float64) * 0.2
        for name, shape in {
            "embedding": (vocab, width), "decay": (width,),
            "recurrent_key": (width, keys), "recurrent_value": (width, width),
            "recurrent_query": (width, keys), "attention_key": (width, keys),
            "attention_value": (width, width), "attention_query": (width, keys),
            "head": (width, vocab),
        }.items():
            self.register_buffer(name, weight(*shape))

    def forward(self, proposed_tokens, initial_state, context_keys,
                context_values, initial_logits):
        x = self.embedding[proposed_tokens]
        a = torch.sigmoid(x @ self.decay)
        k = x @ self.recurrent_key
        v = x @ self.recurrent_value
        q = x @ self.recurrent_query

        attention_queries = x @ self.attention_query
        tentative_keys = torch.cat((context_keys, x @ self.attention_key))
        tentative_values = torch.cat((context_values, x @ self.attention_value))

        state = initial_state.clone()
        snapshots = [state]
        logits = []
        for t in range(proposed_tokens.shape[0]):
            state = a[t] * state + torch.outer(v[t], k[t])
            recurrent_output = state @ q[t]
            end = context_keys.shape[0] + t + 1
            attention_output = causal_attention(
                attention_queries[t], tentative_keys[:end], tentative_values[:end])
            hidden = x[t] + recurrent_output + attention_output
            logits.append(hidden @ self.head)
            snapshots.append(state)

        token_logits = torch.stack(logits)
        accepted_count = accepted_prefix(initial_logits, token_logits, proposed_tokens)
        history = torch.stack(snapshots)
        committed_state = torch.index_select(history, 0, accepted_count).squeeze(0)
        valid_cache_length = context_keys.shape[0] + accepted_count
        return (token_logits, committed_state, tentative_keys, tentative_values,
                valid_cache_length, accepted_count)


def example_inputs():
    generator = torch.Generator().manual_seed(42)
    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)
    return (torch.tensor([1, 2, 3], dtype=torch.int64), randn(4, 3),
            randn(2, 3), randn(2, 4), randn(7))
