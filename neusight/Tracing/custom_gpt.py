import pandas as pd


def _mem_ops(*shapes):
    return [("MEM", tuple(shapes))]


def _vec_row(name, opname, vec_op, input_shapes, output_shape, b, h):
    return {
        "Name": name,
        "OpName": opname,
        "FwOps": [(vec_op, (b, h))],
        "BwOps": [],
        "AccOps": [],
        "InputShapes": input_shapes,
        "OutputShape": output_shape,
    }


def _linear_row(name, b, in_dim, out_dim, output_shape):
    return {
        "Name": name,
        "OpName": "Linear",
        "FwOps": [("Linear", (b, in_dim, out_dim))],
        "BwOps": [],
        "AccOps": [],
        "InputShapes": [(1, 1, in_dim)],
        "OutputShape": output_shape,
    }


def _bmm_row(name, b, m, n, k, input_shapes, output_shape):
    return {
        "Name": name,
        "OpName": "BMM",
        "FwOps": [("BMM", (b, m, n, k))],
        "BwOps": [],
        "AccOps": [],
        "InputShapes": input_shapes,
        "OutputShape": output_shape,
    }


def build_gpt_autoregressive_graph(config, batch_size, sequence_length, execution_type):
    if execution_type not in {"prefill", "decode"}:
        raise ValueError(f"unsupported execution type: {execution_type}")

    hidden = int(config["n_embd"])
    n_head = int(config["n_head"])
    head_dim = hidden // n_head
    vocab = int(config.get("vocab_size", config.get("padded_vocab_size", 50257)))
    ff_dim = hidden * 4

    if execution_type == "prefill":
        query_len = sequence_length
        kv_len = sequence_length
    else:
        query_len = 1
        kv_len = sequence_length

    token_b = batch_size * query_len
    attn_vec_b = batch_size * n_head * query_len
    qk_bmm_b = batch_size * n_head

    rows = [
        {
            "Name": "input_ids",
            "OpName": "misc",
            "FwOps": [],
            "BwOps": [],
            "AccOps": [],
            "InputShapes": [],
            "OutputShape": (batch_size, query_len),
        },
        {
            "Name": "transformer_wte",
            "OpName": "EMBEDDING",
            "FwOps": _mem_ops((batch_size, query_len), (batch_size, query_len, hidden)),
            "BwOps": [],
            "AccOps": [],
            "InputShapes": [(batch_size, query_len)],
            "OutputShape": (batch_size, query_len, hidden),
        },
        {
            "Name": "transformer_wpe",
            "OpName": "EMBEDDING",
            "FwOps": _mem_ops((1, query_len), (1, query_len, hidden)),
            "BwOps": [],
            "AccOps": [],
            "InputShapes": [(1, query_len)],
            "OutputShape": (1, query_len, hidden),
        },
        _vec_row(
            "add_1",
            "VECadd",
            "VECadd",
            [(batch_size, query_len, hidden), (1, query_len, hidden)],
            (batch_size, query_len, hidden),
            token_b,
            hidden,
        ),
        _vec_row(
            "transformer_h_0_ln_1",
            "VECln",
            "VECln",
            [(batch_size, query_len, hidden)],
            (batch_size, query_len, hidden),
            token_b,
            hidden,
        ),
        _linear_row("addmm", token_b, hidden, hidden * 3, (batch_size, query_len, hidden * 3)),
        _bmm_row(
            "matmul",
            qk_bmm_b,
            query_len,
            head_dim,
            kv_len,
            [(batch_size, n_head, query_len, head_dim), (batch_size, n_head, head_dim, kv_len)],
            (batch_size, n_head, query_len, kv_len),
        ),
        _vec_row(
            "truediv",
            "VECdiv",
            "VECdiv",
            [(batch_size, n_head, query_len, kv_len), (1,)],
            (batch_size, n_head, query_len, kv_len),
            attn_vec_b,
            kv_len,
        ),
        _vec_row(
            "softmax",
            "VECsoftmax",
            "VECsoftmax",
            [(batch_size, n_head, query_len, kv_len)],
            (batch_size, n_head, query_len, kv_len),
            attn_vec_b,
            kv_len,
        ),
        _bmm_row(
            "matmul_1",
            qk_bmm_b,
            query_len,
            kv_len,
            head_dim,
            [(batch_size, n_head, query_len, kv_len), (batch_size, n_head, kv_len, head_dim)],
            (batch_size, n_head, query_len, head_dim),
        ),
        _linear_row("addmm_1", token_b, hidden, hidden, (batch_size, query_len, hidden)),
        _vec_row(
            "add_8",
            "VECadd",
            "VECadd",
            [(batch_size, query_len, hidden), (batch_size, query_len, hidden)],
            (batch_size, query_len, hidden),
            token_b,
            hidden,
        ),
        _vec_row(
            "transformer_h_0_ln_2",
            "VECln",
            "VECln",
            [(batch_size, query_len, hidden)],
            (batch_size, query_len, hidden),
            token_b,
            hidden,
        ),
        _linear_row("addmm_2", token_b, hidden, ff_dim, (batch_size, query_len, ff_dim)),
        _vec_row(
            "gelu",
            "VECgelu",
            "VECgelu",
            [(batch_size, query_len, ff_dim)],
            (batch_size, query_len, ff_dim),
            token_b,
            ff_dim,
        ),
        _linear_row("addmm_3", token_b, ff_dim, hidden, (batch_size, query_len, hidden)),
        _vec_row(
            "add_15",
            "VECadd",
            "VECadd",
            [(batch_size, query_len, hidden), (batch_size, query_len, hidden)],
            (batch_size, query_len, hidden),
            token_b,
            hidden,
        ),
        _vec_row(
            "transformer_ln_f",
            "VECln",
            "VECln",
            [(batch_size, query_len, hidden)],
            (batch_size, query_len, hidden),
            token_b,
            hidden,
        ),
        _linear_row("lm_head", token_b, hidden, vocab, (batch_size, query_len, vocab)),
    ]

    return pd.DataFrame(rows)
