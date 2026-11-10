#!/usr/bin/env python3
import argparse
import sys

import torch
import torch.nn.functional as F

from bio.model import DDSMScoreAdapter, PromoterCNN
from bio.processes import (
    d3pm_training_batch,
    dirichlet_fm_training_batch,
    discrete_fm_training_batch,
    sample_d3pm,
    sample_dirichlet_fm,
    sample_discrete_fm,
)


parser = argparse.ArgumentParser()
parser.add_argument("--ddsm-upstream", required=True)
args = parser.parse_args()
torch.manual_seed(7)
batch, length = 2, 16
tokens = torch.randint(4, (batch, length))
signal = torch.rand(batch, length, 1)

for name, training, sampler in [
    ("d3pm", lambda: d3pm_training_batch(tokens, 4), lambda model: sample_d3pm(model, signal, 4)),
    ("discrete_fm", lambda: discrete_fm_training_batch(tokens), lambda model: sample_discrete_fm(model, signal, 4)),
]:
    model = PromoterCNN(4, channels=16, layers=2)
    state, time = training()
    loss = F.cross_entropy(model(state, signal, time).transpose(1, 2), tokens)
    loss.backward()
    samples = sampler(model)
    assert samples.shape == tokens.shape and samples.min() >= 0 and samples.max() < 4
    print(name, "PASS", float(loss))

model = PromoterCNN(8, channels=16, layers=2)
state, time = dirichlet_fm_training_batch(tokens, 2.0, 8.0, 2.0)
loss = F.cross_entropy(model(state, signal, time).transpose(1, 2), tokens)
loss.backward()
samples, diagnostics = sample_dirichlet_fm(model, signal, 4, 8.0, 2.0)
assert samples.shape == tokens.shape and samples.min() >= 0 and samples.max() < 4
print("dirichlet_fm", "PASS", float(loss), diagnostics)

sys.path.insert(0, args.ddsm_upstream)
import ddsm

model = PromoterCNN(4, channels=16, layers=2, center_output=True)
adapter = DDSMScoreAdapter(model)
simplex = torch.distributions.Dirichlet(torch.ones(4)).sample((batch, length))
score = adapter(torch.cat([simplex, signal], -1), torch.rand(batch))
assert torch.allclose(score.sum(-1), torch.zeros(batch, length), atol=1e-5)
stick_score = ddsm.gx_to_gv(score, simplex, create_graph=True)
assert stick_score.shape == (batch, length, 3) and torch.isfinite(stick_score).all()
stick_score.square().mean().backward()
print("ddsm_core", "PASS")
print("bio_core_selftest=PASS")
