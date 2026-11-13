#!/usr/bin/env python
"""Domain-specialised expert extension for ibm-granite/granite-3.0-1b-a400m-base.

Pipeline (script version of granite-extended.ipynb):
  1. Load the base model and keep an untouched copy of its weights for the final model.
  2. Load + preprocess the three domain datasets (math / code / creative) with a train-test split.
  3. Per domain: extend every MoE layer by EXPERTS_PER_DOMAIN experts, train ONLY these experts and
     ONLY their router rows (all other gradients are zeroed in the backward pass).
  4. Merge: append the trained experts and router rows of all domains to the base model.
  5. Run the final model on the test splits and log, per token, the source dataset and the experts
     it was routed to in every layer.

Usage:
  python granite_extended.py --out-dir runs/granite-extended
  # quick smoke test:
  python granite_extended.py --out-dir runs/smoke --n-train 16 --n-test 4 --max-steps 2 --max-length 128
"""

import argparse
import json
import os
import re

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

BASE = "ibm-granite/granite-3.0-1b-a400m-base"
EXPERTS_PER_DOMAIN = 4
SEED = 0

# Parameters that carry one row per expert (dim 0 = expert index). Everything else is frozen.
MOE_KEY = re.compile(r"^model\.layers\.\d+\.block_sparse_moe\.(router\.weight|experts\.gate_up_proj|experts\.down_proj)$")


# --------------------------------------------------------------------------------------------------
# Datasets (identical preprocessing to the notebook)
# --------------------------------------------------------------------------------------------------
WP = re.compile(r"^\s*\[\s*WP\s*\]", re.IGNORECASE)  # filter: must start with [ WP ]
TAGS = re.compile(r"^(\s*\[[^\]]*\]\s*)+")  # strip: all leading [ .. ] tags


def clean(s):
    return s.replace(" <newline> ", "\n").replace("<newline>", "\n").strip()


def prep(name, p, c, n, n_test, fmt=lambda x: x, cfmt=lambda x: x, keep=None, split="train"):
    ds = load_dataset(name, split=split)
    if keep is not None:
        ds = ds.filter(keep)
    ds = ds.shuffle(seed=0).select(range(min(n + n_test, len(ds))))
    ds = ds.map(lambda ex: {"prompt": fmt(ex[p]) + "\n\n", "completion": cfmt(ex[c])},
                remove_columns=ds.column_names)
    return ds.train_test_split(test_size=n_test, shuffle=False)  # already shuffled above


def load_domains(n, n_test):
    domains = {
        "math": prep("meta-math/MetaMathQA", "query", "response", n, n_test,
                     fmt=lambda q: f"Question: {q}\nAnswer:"),
        "code": prep("ise-uiuc/Magicoder-OSS-Instruct-75K", "problem", "solution", n, n_test),
        "creative": prep("euclaise/writingprompts", "prompt", "story", n, n_test,
                         keep=lambda ex: WP.match(ex["prompt"]) is not None,
                         fmt=lambda s: clean(TAGS.sub("", s)),
                         cfmt=clean),
    }
    for split in ("train", "test"):
        bad = [x for x in domains["creative"][split]["prompt"] if x.lstrip().startswith("[")]
        assert not bad, f"{len(bad)} tagged creative prompts left in {split}, e.g. {bad[:3]}"
    for name, dd in domains.items():
        print(f"[data] {name:9s} train={len(dd['train']):,} test={len(dd['test']):,}")
    return domains


# --------------------------------------------------------------------------------------------------
# Model surgery
# --------------------------------------------------------------------------------------------------
def build_model(state_dict, n_experts, dtype):
    cfg = AutoConfig.from_pretrained(BASE)
    cfg.num_local_experts = n_experts
    cfg.output_router_logits = False
    model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    model.load_state_dict(state_dict, strict=True)
    return model


def moe_keys(state_dict):
    return sorted(k for k in state_dict if MOE_KEY.match(k))


def extend_state_dict(base_sd, n_old, n_new, generator):
    """Append n_new experts to every MoE layer. Each new expert (and its router row) starts as a
    noisy copy of a randomly chosen old expert; the same source expert is used for the router row,
    gate_up_proj and down_proj so that the copied expert is internally consistent."""
    src = torch.randint(0, n_old, (n_new,), generator=generator)
    new_sd = {}
    for k, o in base_sd.items():
        if MOE_KEY.match(k):
            assert o.shape[0] == n_old, f"{k}: expected {n_old} experts in dim 0, got {tuple(o.shape)}"
            extra = o[src].float().clone()
            noise = torch.randn(extra.shape, generator=generator)
            extra += 0.01 * o.float().std() * noise  # break symmetry with the source expert
            new_sd[k] = torch.cat([o, extra.to(o.dtype)], 0)
        else:
            new_sd[k] = o
    return new_sd, src


def freeze_all_but_new_experts(model, n_old):
    """requires_grad only on expert/router tensors, and a hook that zeroes the gradient of the
    first n_old rows (old experts / their router rows) during the backward pass."""
    for p in model.parameters():
        p.requires_grad_(False)

    def only_new(grad):
        grad = grad.clone()
        grad[:n_old] = 0
        return grad

    for layer in model.model.layers:
        moe = layer.block_sparse_moe
        for p in (moe.experts.gate_up_proj, moe.experts.down_proj, moe.router.weight):
            p.requires_grad_(True)
            p.register_hook(only_new)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_effective = sum(p[n_old:].numel() for p in model.parameters() if p.requires_grad)
    print(f"[freeze] tensors with grad: {n_trainable:,} params, effectively trained (new rows): {n_effective:,}")


@torch.no_grad()
def verify_untouched(model, base_sd, n_old):
    """Assert that every base weight (incl. old expert/router rows) is bit-identical to the base."""
    sd = model.state_dict()
    for k, b in base_sd.items():
        cur = sd[k].detach().cpu()
        if MOE_KEY.match(k):
            cur = cur[:n_old]
        if not torch.equal(cur, b.to(cur.dtype)):
            raise RuntimeError(f"base weight changed during training: {k}")
    print("[verify] all base weights (incl. old experts / router rows) unchanged")


# --------------------------------------------------------------------------------------------------
# Training one domain
# --------------------------------------------------------------------------------------------------
def train_domain(name, idx, train_ds, base_sd, n_old, tok, args):
    rows_path = os.path.join(args.out_dir, "domain_experts", f"{name}.pt")
    if os.path.exists(rows_path) and not args.retrain:
        print(f"[train:{name}] found {rows_path}, skipping training (use --retrain to force)")
        return torch.load(rows_path)

    set_seed(SEED + idx)
    gen = torch.Generator().manual_seed(SEED + idx)
    sd, src = extend_state_dict(base_sd, n_old, EXPERTS_PER_DOMAIN, gen)
    print(f"[train:{name}] new experts initialised from old experts {src.tolist()}")

    # fp32 master weights (bf16 autocast for compute): pure-bf16 weights would round away most
    # Adam updates of size ~lr relative to weights of size ~1e-2.
    model = build_model(sd, n_old + EXPERTS_PER_DOMAIN, torch.float32)
    del sd
    # TRL's fused-LM-head forward computes the load-balancing loss with `config.num_experts`, which
    # GraniteMoe calls `num_local_experts`.
    model.config.num_experts = model.config.num_local_experts
    freeze_all_but_new_experts(model, n_old)

    cfg = SFTConfig(
        output_dir=os.path.join(args.out_dir, "trainer", name),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.0,  # decoupled weight decay would otherwise shrink the frozen rows too
        num_train_epochs=1,
        max_steps=args.max_steps,
        max_length=args.max_length,
        packing=True,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # needed because the embeddings are frozen
        router_aux_loss_coef=0.01,  # load balancing so the new experts actually get tokens
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=SEED + idx,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, processing_class=tok)
    trainer.train()

    model = trainer.model
    verify_untouched(model, base_sd, n_old)
    msd = model.state_dict()
    rows = {k: msd[k][n_old:].detach().cpu().clone() for k in moe_keys(base_sd)}
    os.makedirs(os.path.dirname(rows_path), exist_ok=True)
    torch.save(rows, rows_path)
    print(f"[train:{name}] saved trained expert/router rows to {rows_path}")

    del trainer, model, msd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------------------------------
def merge(final_sd, domain_rows, n_old):
    keys = moe_keys(final_sd)
    groups = {"base": list(range(n_old))}
    offset = n_old
    for name, rows in domain_rows.items():
        assert set(rows) == set(keys), f"{name}: trained rows do not cover all MoE tensors"
        groups[name] = list(range(offset, offset + EXPERTS_PER_DOMAIN))
        offset += EXPERTS_PER_DOMAIN
    for k in keys:
        base = final_sd[k]
        parts = [base]
        for name, rows in domain_rows.items():
            r = rows[k]
            assert r.shape[0] == EXPERTS_PER_DOMAIN and r.shape[1:] == base.shape[1:], (k, name, r.shape)
            parts.append(r.to(base.dtype))
        final_sd[k] = torch.cat(parts, 0)
        assert final_sd[k].shape[0] == offset
    return final_sd, groups, offset


# --------------------------------------------------------------------------------------------------
# Routing analysis
# --------------------------------------------------------------------------------------------------
def tokenize_like_trl(tok, prompt, completion, max_length):
    """Same tokenization as SFTTrainer for non-conversational prompt-completion data."""
    if not completion.endswith(tok.eos_token):
        completion = completion + tok.eos_token
    prompt_ids = tok(prompt)["input_ids"]
    ids = tok(prompt + completion)["input_ids"]
    prompt_len = 0
    for a, b in zip(prompt_ids, ids):
        if a != b:
            break
        prompt_len += 1
    ids = ids[:max_length]
    return ids, min(prompt_len, len(ids))


@torch.inference_mode()
def log_routing(model, tok, domains, groups, n_experts, args):
    device = next(model.parameters()).device
    layers = model.model.layers
    n_layers, top_k = len(layers), model.config.num_experts_per_tok
    assert n_experts <= 256, "expert ids are stored as uint8"

    captured = [None] * n_layers

    def make_hook(i):
        def hook(module, inputs, output):
            _, top_k_weights, top_k_index = output  # (bsz * seq_len, top_k) each
            captured[i] = (top_k_index, top_k_weights)
        return hook

    handles = [layer.block_sparse_moe.router.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]

    expert_group = torch.empty(n_experts, dtype=torch.long)
    group_names = list(groups)
    for gi, g in enumerate(group_names):
        expert_group[groups[g]] = gi

    log, summary = {}, {}
    try:
        for name, dd in domains.items():
            test = dd["test"]
            encoded = [tokenize_like_trl(tok, ex["prompt"], ex["completion"], args.max_length) for ex in test]
            order = sorted(range(len(encoded)), key=lambda i: -len(encoded[i][0]))  # length-sorted batching

            rec = {k: [] for k in ("sample_idx", "position", "token_id", "is_completion", "experts", "weights")}
            for start in range(0, len(order), args.eval_batch_size):
                batch_idx = order[start:start + args.eval_batch_size]
                lens = [len(encoded[i][0]) for i in batch_idx]
                L = max(lens)
                input_ids = torch.full((len(batch_idx), L), tok.pad_token_id, dtype=torch.long)
                attn = torch.zeros((len(batch_idx), L), dtype=torch.long)
                for b, i in enumerate(batch_idx):  # right padding: real tokens are unaffected (causal)
                    input_ids[b, :lens[b]] = torch.tensor(encoded[i][0])
                    attn[b, :lens[b]] = 1

                captured[:] = [None] * n_layers
                model(input_ids=input_ids.to(device), attention_mask=attn.to(device), use_cache=False)
                assert all(c is not None for c in captured)

                idx = torch.stack([c[0] for c in captured], 1).view(len(batch_idx), L, n_layers, top_k).cpu()
                w = torch.stack([c[1] for c in captured], 1).view(len(batch_idx), L, n_layers, top_k).float().cpu()
                for b, i in enumerate(batch_idx):
                    n_tok, p_len = lens[b], encoded[i][1]
                    rec["sample_idx"].append(torch.full((n_tok,), i, dtype=torch.int32))
                    rec["position"].append(torch.arange(n_tok, dtype=torch.int32))
                    rec["token_id"].append(input_ids[b, :n_tok].to(torch.int32))
                    rec["is_completion"].append(torch.arange(n_tok) >= p_len)
                    rec["experts"].append(idx[b, :n_tok].to(torch.uint8))
                    rec["weights"].append(w[b, :n_tok].to(torch.float16))
            rec = {k: torch.cat(v) for k, v in rec.items()}
            log[name] = rec

            # ---- summary statistics ----
            experts = rec["experts"].long()  # (T, n_layers, top_k)
            counts = torch.zeros(n_layers, n_experts, dtype=torch.long)
            for l in range(n_layers):
                counts[l] = torch.bincount(experts[:, l].reshape(-1), minlength=n_experts)
            grp = expert_group[experts]  # (T, n_layers, top_k)

            def group_share(mask):
                g = grp[mask]
                if g.numel() == 0:
                    return {}
                c = torch.bincount(g.reshape(-1), minlength=len(group_names)).double()
                return {gn: float(c[gi] / c.sum()) for gi, gn in enumerate(group_names)}

            per_layer = []
            for l in range(n_layers):
                c = torch.bincount(grp[:, l].reshape(-1), minlength=len(group_names)).double()
                per_layer.append({gn: float(c[gi] / c.sum()) for gi, gn in enumerate(group_names)})
            own = groups[name]
            hits_own = torch.isin(experts, torch.tensor(own)).any(-1)  # (T, n_layers)
            all_mask = torch.ones(len(experts), dtype=torch.bool)
            summary[name] = {
                "n_samples": len(test),
                "n_tokens": int(len(experts)),
                "n_completion_tokens": int(rec["is_completion"].sum()),
                "group_share_all_tokens": group_share(all_mask),
                "group_share_prompt_tokens": group_share(~rec["is_completion"]),
                "group_share_completion_tokens": group_share(rec["is_completion"]),
                "group_share_per_layer": per_layer,
                "frac_tokens_with_own_expert_per_layer": hits_own.float().mean(0).tolist(),
                "expert_counts_per_layer": counts.tolist(),
            }
    finally:
        for h in handles:
            h.remove()

    return log, summary, group_names


def print_summary(summary, groups):
    group_names = list(groups)
    print("\n[routing] share of top-k routing slots per expert group (all layers, all tokens)")
    header = f"{'tokens from':12s}" + "".join(f"{g:>10s}" for g in group_names)
    print(header)
    print("-" * len(header))
    for name, s in summary.items():
        print(f"{name:12s}" + "".join(f"{s['group_share_all_tokens'][g]:10.3f}" for g in group_names))
    print("\n[routing] same, completion tokens only")
    print(header)
    print("-" * len(header))
    for name, s in summary.items():
        sh = s["group_share_completion_tokens"]
        print(f"{name:12s}" + "".join(f"{sh.get(g, float('nan')):10.3f}" for g in group_names))
    n_total = sum(len(v) for v in groups.values())
    print("\n(uniform routing would give " + ", ".join(f"{g}={len(v) / n_total:.3f}" for g, v in groups.items()) + ")")


# --------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="runs/granite-extended")
    ap.add_argument("--n-train", type=int, default=40_000, help="training samples per domain")
    ap.add_argument("--n-test", type=int, default=500, help="test samples per domain")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=-1, help="-1 = one full epoch")
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--retrain", action="store_true", help="ignore saved per-domain expert rows")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(SEED)

    # 1. base model + untouched copy of its weights for the final model
    tok = AutoTokenizer.from_pretrained(BASE)
    base_model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    n_old = base_model.config.num_local_experts  # 32
    base_sd = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    del base_model
    final_sd = {k: v.clone() for k, v in base_sd.items()}  # copy for the final model
    assert len(moe_keys(base_sd)) == 3 * AutoConfig.from_pretrained(BASE).num_hidden_layers # 3 because router + gate_up_proj + down_proj

    # 2. data (+ train/test split)
    domains = load_domains(args.n_train, args.n_test)

    # 3. train EXPERTS_PER_DOMAIN new experts per domain
    domain_rows = {}
    for idx, (name, dd) in enumerate(domains.items()):
        domain_rows[name] = train_domain(name, idx, dd["train"], base_sd, n_old, tok, args)

    # 4. merge into the final model
    final_sd, groups, n_total = merge(final_sd, domain_rows, n_old)
    final_model = build_model(final_sd, n_total, torch.bfloat16)
    del final_sd
    final_dir = os.path.join(args.out_dir, "final_model")
    final_model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    with open(os.path.join(args.out_dir, "expert_groups.json"), "w") as f:
        json.dump(groups, f, indent=2)
    print(f"[merge] final model with {n_total} experts/layer saved to {final_dir}; groups: "
          + ", ".join(f"{g}={v[0]}..{v[-1]}" for g, v in groups.items()))

    # 5. routing behaviour on the test splits
    device = "cuda" if torch.cuda.is_available() else "cpu"
    final_model.to(device).eval()
    log, summary, group_names = log_routing(final_model, tok, domains, groups, n_total, args)
    torch.save({
        "description": "Per-token routing on the test splits. For each domain (= source dataset): "
                       "sample_idx [T], position [T], token_id [T], is_completion [T], "
                       "experts [T, n_layers, top_k] (uint8 expert ids), weights [T, n_layers, top_k] (fp16).",
        "expert_groups": groups,
        "n_layers": final_model.config.num_hidden_layers,
        "top_k": final_model.config.num_experts_per_tok,
        "domains": log,
    }, os.path.join(args.out_dir, "routing_log.pt"))
    with open(os.path.join(args.out_dir, "routing_summary.json"), "w") as f:
        json.dump({"expert_groups": groups, "domains": summary}, f, indent=2)
    print_summary(summary, groups)
    print(f"\n[done] routing log: {os.path.join(args.out_dir, 'routing_log.pt')}, "
          f"summary: {os.path.join(args.out_dir, 'routing_summary.json')}")


if __name__ == "__main__":
    main()
