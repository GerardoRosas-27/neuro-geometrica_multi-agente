import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from tc_common import ByteTok, build_conv, pool, boot, ctx_features, asst_positions, derangement, load_model, state_dict_sha
from turno_encoder import fit, encode
from tc_decode import Adapter, freeze, emb_rms, examples, response_nll

MSG = [{"role": "user", "content": "Hola"}, {"role": "assistant", "content": "Hola, ¿en qué ayudo?"},
       {"role": "user", "content": "Dime algo"}, {"role": "assistant", "content": "Algo."}]


class T(unittest.TestCase):
    def test_spans_exactos(self):
        tok = ByteTok(); ids, turns = build_conv(tok, MSG)
        self.assertEqual(len(turns), 4)
        for tu, m in zip(turns, MSG):
            self.assertEqual(tok.decode(ids[tu["content"][0]:tu["content"][1]]), " " + m["content"] + "\n")
            self.assertTrue(tok.decode(ids[tu["hdr"][0]:tu["hdr"][1]]).endswith(":"))
        self.assertEqual(turns[0]["hdr"][0], 1)  # tras BOS

    def test_pool(self):
        H = np.arange(12.0).reshape(6, 2)
        np.testing.assert_allclose(pool(H, (1, 3)), H[1:3].mean(0)); np.testing.assert_allclose(pool(H, (1, 3), "last"), H[2])

    def test_pca_blanqueado(self):
        rng = np.random.default_rng(0); H = rng.normal(size=(500, 10)) @ rng.normal(size=(10, 10))
        enc = fit(H, 4); Z = encode(H, enc)
        np.testing.assert_allclose(Z.mean(0), 0, atol=1e-5); np.testing.assert_allclose(Z.var(0, ddof=1), 1, rtol=1e-3)

    def test_boot(self):
        r = boot(np.ones(30) * 2, np.ones(30)); self.assertAlmostEqual(r["media"], 1.0); self.assertEqual(r["ic95"], [1.0, 1.0])

    def test_ctx_features_y_posiciones(self):
        Z = np.arange(10.0).reshape(5, 2)
        np.testing.assert_allclose(ctx_features(Z, 3, 4), np.r_[Z[2], Z[1], Z[0], Z[0]])
        self.assertEqual(asst_positions(["user", "assistant", "user", "assistant"]), [1, 3])
        p = derangement(7); self.assertFalse((p == np.arange(7)).any())

    def test_examples_causales(self):
        tok = ByteTok(); ids, turns = build_conv(tok, MSG)
        ex = examples([{"ids": ids, "turns": turns}], 384, 1)
        self.assertEqual([e[1] for e in ex], [1, 3])
        for c, t, ctx, resp in ex:
            self.assertTrue(tok.decode(ctx).endswith("Assistant:")); self.assertNotIn(MSG[t]["content"], tok.decode(ctx))

    def test_identidad_y_llm_congelado(self):
        m, tok, mm = load_model(tiny_random=True); freeze(m); sha = state_dict_sha(m)
        ids, turns = build_conv(tok, MSG); (_, _, ctx, resp) = examples([{"ids": ids, "turns": turns}], 384, 1)[0]
        a = float(response_nll(m, ctx, resp)); b = float(response_nll(m, ctx, resp, use_embeds=False))
        self.assertLess(abs(a - b), 1e-4)
        ad = Adapter(8, mm["hidden"], 4, 32, emb_rms(m)); opt = torch.optim.SGD(ad.parameters(), 0.1)
        l0 = None
        for _ in range(5):
            opt.zero_grad(); l = response_nll(m, ctx, resp, ad(torch.ones(1, 8))); l.backward(); opt.step(); l0 = l0 or float(l)
        self.assertLess(float(response_nll(m, ctx, resp, ad(torch.ones(1, 8)))), l0)  # el adaptador aprende
        self.assertEqual(state_dict_sha(m), sha)  # LLM intacto
        self.assertTrue(all(p.grad is None for p in m.parameters()))


if __name__ == "__main__":
    unittest.main()
