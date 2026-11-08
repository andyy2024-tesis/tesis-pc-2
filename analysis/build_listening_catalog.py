"""Single catalog of the models on the listening page; writes it into the page's MODELS constant.

Each entry declares what differs from its parent. `diff` is checked against the saved configs (config.json for runs trained
from scratch, meta.json for adversarial fine-tunes), so a wrong description stops the build. Loss and wind come from
analysis/results/catalog_metrics.json (analysis/measure_catalog.py). Audio files keep their names: `key` is the file name.

Relations: "rec" = same recipe plus a change, trained from zero; "cont" = continues from the parent's weights.
Usage: uv run python analysis/build_listening_catalog.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "analysis" / "results"
PAGE = RES / "slides" / "listen" / "index.html"
V5 = "e6/structured_v5_body_audio"
ADV = dict(steps=3000, d_ch=64, d_lr_mult=2.0)  # the adversarial recipe of the favorite
B6K = dict(body_ir_cutoff=6000.0, body_eq_max=8000.0, body_eq_bands=10)
MINUS = object()  # the setting was removed from the recipe

# In training order (parents first). run: results folder; key: file name when it isn't derived from run.
C = [
    dict(n="S1 Estructurado", run="structured_v2", change="Primer modelo estructurado: parciales de cuerda rígida, plantilla por nota, decaimiento físico y corrección de la GRU. Solo partitura."),
    dict(n="S1 + cuerpo", run="structured_v2_bodyir", p="structured_v2", rel="rec", diff=dict(body_ir=True), change="Respuesta del cuerpo excitada por cada pulsación, hasta 600 Hz."),
    dict(n="S1 + cuerpo · entrenado sin ruido", run="structured_v2_bodyir_nonoise", p="structured_v2_bodyir", rel="rec", diff=dict(noise=False), change="Se entrena sin el ruido residual."),
    dict(n="D1 DDSP", run="ddsp_v2", change="DDSP simple: armónicos más ruido, sin estructura. Solo partitura."),
    dict(n="D1 · entrenado sin ruido", run="ddsp_v2_nonoise", p="ddsp_v2", rel="rec", diff=dict(noise=False), change="Se entrena sin el ruido residual."),
    dict(n="S2", run="structured_v4_body", p="structured_v2_bodyir", rel="rec", diff={}, code_only=True, change="Ajustes por grabación en tablas propias (cuerpo, piso de ruido, transitorio), con 10× de learning rate. La config no lo muestra: viene del código (E4, ronda 3) y se confirma en los pesos."),
    dict(n="S2 + reverb", run="structured_v4_body_reverb", p="structured_v4_body", rel="rec", diff=dict(reverb=True), change="Reverb de sala por grabación."),
    dict(n="M1 Base", run="structured_v5_body_audio", p="structured_v4_body", rel="rec", diff=dict(audio_feats=True), change="Features de audio: loudness y pitch de cada nota cada 10 ms (resíntesis)."),
    dict(n="D2", run="ddsp_v5_audio", p="ddsp_v2", rel="rec", diff=dict(audio_feats=True), change="Las mismas features de audio."),
    dict(n="M1 sin ruido + adv", run=f"{V5}_nonoise_strongD", p="structured_v5_body_audio", rel="cont", meta=dict(overrides=dict(noise=False)), meta_note=True, change="Adversarial de 3000 pasos, quitándole el ruido. Los pasos y el discriminator vienen de E6_findings: este meta.json no los guarda."),
    dict(n="M1 · loss mel", run="structured_v6_mel_noise", p="structured_v5_body_audio", rel="rec", diff={}, loss="mel", change="Se entrena con loss mel en vez de la espectral lineal + log."),
    dict(n="M1 · loss mel, entrenado sin ruido", run="structured_v6_mel_nonoise", p="structured_v6_mel_noise", rel="rec", diff=dict(noise=False), loss="mel", change="Se entrena sin el ruido residual."),
    dict(n="M1 sin ruido + adv · 2000 pasos", key="adv9k-step2000", p="structured_v5_body_audio", rel="cont", fixed=dict(mean=6.79, wind=None), change="Paso 2000 de la corrida de 9000. Esos pesos no se guardaron: la loss viene de E6_findings y no se volvió a medir."),
    dict(n="M1 sin ruido + adv · 9000 pasos", run=f"{V5}_nonoise_strongD_9k", p="structured_v5_body_audio", rel="cont", meta=dict(overrides=dict(noise=False), steps=9000, d_ch=64, d_lr_mult=2.0), change="Reiniciado desde M1 con los mismos ajustes, 9000 pasos."),
    dict(n="M1 + adv", run=f"{V5}_noise_strongD", p="structured_v5_body_audio", rel="cont", meta=dict(overrides={}, **ADV), change="Adversarial de 3000 pasos, conservando el ruido."),
    dict(n="M1 + adv · loss solo lineal", run=f"{V5}_noise_strongD_linonly", p="structured_v5_body_audio", rel="cont", meta=dict(overrides={}, spec_log_weight=0.0, **ADV), change="Como M1 + adv, pero la loss espectral no tiene el término log."),
    dict(n="M1 + adv · 9000 pasos", run=f"{V5}_noise_strongD_9k", p=f"{V5}_noise_strongD", rel="cont", meta=dict(overrides={}, steps=9000, d_ch=64, d_lr_mult=2.0), change="Sigue hasta 9000 pasos."),
    dict(n="M1 + simpática", run="structured_v8_symp", p="structured_v5_body_audio", rel="rec", diff=dict(sympathetic=True), change="Las cuerdas al aire resuenan con los parciales cercanos."),
    dict(n="M1 + clic", run="structured_v8_click", p="structured_v5_body_audio", rel="rec", diff=dict(pluck_click=True), change="Forma de onda aprendida de uña y dedo en cada ataque."),
    dict(n="M1 + cuerpo 6 kHz", run="structured_v8_bodyhi", p="structured_v5_body_audio", rel="rec", diff=B6K, change="Respuesta del cuerpo hasta 6 kHz (antes 600 Hz) y EQ hasta 8 kHz."),
    dict(n="M1 + squeak", run="structured_v8_squeak", p="structured_v5_body_audio", rel="rec", diff=dict(squeak=True), change="Ruido justo antes de las notas en las cuerdas entorchadas."),
    dict(n="M1 + dos decaimientos", run="structured_v8_twodecay", p="structured_v5_body_audio", rel="rec", diff=dict(two_decay=True), change="Dos polarizaciones por parcial, cada una con su decaimiento."),
    dict(n="M1 + adv · D complejo + log", run=f"{V5}_noise_strongD_both", p="structured_v5_body_audio", rel="cont", meta=dict(overrides={}, d_kind="both", **ADV), change="Como M1 + adv, con un discriminator que ve el complejo y la magnitud log."),
    dict(n="M2 Combinado", run="structured_v8_combo", p="structured_v5_body_audio", rel="rec", diff=dict(two_decay=True, pluck_click=True, **B6K), change="Junta cuerpo 6 kHz, dos decaimientos y clic."),
    dict(n="M1 + adv · D solo log", run=f"{V5}_noise_strongD_logmag", p="structured_v5_body_audio", rel="cont", meta=dict(overrides={}, d_kind="logmag", **ADV), change="Como M1 + adv, con un discriminator que solo ve la magnitud log."),
    dict(n="M2 + adv · D complejo + log", run="e6/structured_v8_combo_noise_strongD_both", p="structured_v8_combo", rel="cont", meta=dict(overrides={}, d_kind="both", **ADV), change="Adversarial de 3000 pasos, discriminator complejo + log."),
    dict(n="M2 · entrenado sin ruido", run="structured_v8_combo_nonoise", p="structured_v8_combo", rel="rec", diff=dict(noise=False), change="Se entrena sin el ruido residual. No es lo mismo que M2 con el ruido apagado."),
    dict(n="M1 + compuerta", run="structured_v5_gated", p="structured_v5_body_audio", rel="rec", diff=dict(noise_gate=True), change="El ruido solo suena alrededor de cada ataque."),
    dict(n="M2 + compuerta", run="structured_v8_gated", p="structured_v8_combo", rel="rec", diff=dict(noise_gate=True), change="El ruido solo suena alrededor de cada ataque."),
    dict(n="M1 + compuerta + adv", run="e6/structured_v5_gated_noise_strongD", p="structured_v5_gated", rel="cont", meta=dict(overrides={}, d_kind="complex", **ADV), change="Adversarial de 3000 pasos."),
    dict(n="M2 + compuerta + timbre 16", run="structured_v9_timbre", p="structured_v8_gated", rel="rec", diff=dict(note_timbre=True), change="La red de cada nota lee del audio el timbre del ataque: 16 armónicos."),
    dict(n="M2 + compuerta + adv", run="e6/structured_v8_gated_noise_strongD", p="structured_v8_gated", rel="cont", meta=dict(overrides={}, d_kind="complex", **ADV), change="Adversarial de 3000 pasos."),
    dict(n="M2 + compuerta + timbre 32", run="structured_v9_timbre32", p="structured_v9_timbre", rel="rec", diff=dict(timbre_k=32), change="Lee 32 armónicos en vez de 16."),
    dict(n="M2 + compuerta + timbre 16 + adv", run="e6/structured_v9_timbre_noise_strongD", p="structured_v9_timbre", rel="cont", meta=dict(overrides={}, d_kind="complex", **ADV), change="Adversarial de 3000 pasos."),
    dict(n="M2 + compuerta + timbre 32 + adv", run="e6/structured_v9_timbre32_noise_strongD", p="structured_v9_timbre32", rel="cont", meta=dict(overrides={}, d_kind="complex", **ADV), change="Adversarial de 3000 pasos."),
    dict(n="M2 + compuerta + timbre 32 + adv · 9000 pasos", run="e6/structured_v9_timbre32_noise_strongD_9k", p="e6/structured_v9_timbre32_noise_strongD", rel="cont", meta=dict(overrides={}, d_kind="complex", steps=9000, d_ch=64, d_lr_mult=2.0), change="Sigue hasta 9000 pasos."),
    dict(n="M2 + compuerta + timbre 32 + lectura tardía", run="structured_v9_timbre32late", p="structured_v9_timbre32", rel="rec", diff=dict(timbre_late=True), change="También lee los armónicos 150–235 ms después del ataque."),
    dict(n="M2 + timbre 32", run="structured_v9_t32_no_gate", p="structured_v8_combo", rel="rec", diff=dict(note_timbre=True, timbre_k=32, noise_gate=False), live=True, change="Timbre de 32 armónicos, con el ruido sonando todo el tiempo."),
    dict(n="M1 + cuerpo 6 kHz + compuerta + timbre 32", run="structured_v9_t32_simple", p="structured_v9_timbre32", rel="rec", diff=dict(two_decay=MINUS, pluck_click=MINUS), change="M2 + compuerta + timbre 32, sin dos decaimientos y sin clic."),
    dict(n="M1 + timbre 32", run="structured_v9_base_t32", p="structured_v5_body_audio", rel="rec", diff=dict(note_timbre=True, timbre_k=32), live=True, change="Timbre de 32 armónicos sobre la base, con el ruido sonando todo el tiempo."),
    dict(n="M1 + timbre 32 + adv", run="e6/structured_v9_base_t32_noise_strongD", p="structured_v9_base_t32", rel="cont", meta=dict(overrides={}, d_kind="complex", **ADV), live=True, fav=True, change="Adversarial de 3000 pasos. Es M1 + adv con el timbre como único cambio."),
    dict(n="M2 + timbre 32 + adv", run="e6/structured_v9_t32_no_gate_noise_strongD", p="structured_v9_t32_no_gate", rel="cont", meta=dict(overrides={}, d_kind="complex", **ADV), live=True, change="Adversarial de 3000 pasos."),
]


def file_key(e):
    return e.get("key") or e["run"].replace("/", "-")


def overrides(run):
    return json.loads((RES / "e4" / run / "config.json").read_text())["overrides"]


def check(e, finished):
    """Raise if the entry disagrees with the saved configs."""
    run, p = e.get("run"), e.get("p")
    if run is None or not finished:
        return
    if run.startswith("e6/"):
        meta = json.loads((RES / run / "meta.json").read_text())
        args = meta.get("args", {})
        root = p
        while root.startswith("e6/"):  # a continuation records the run its chain started from
            root = next(x for x in C if x.get("run") == root)["p"]
        assert meta["base"] == root, f"{run}: meta base {meta['base']} != {root}"
        for k, v in e["meta"].items():
            got = meta.get("overrides") if k == "overrides" else args.get(k, "complex" if k == "d_kind" else None)
            assert got == v, f"{run}: {k} is {got}, catalog says {v}"
        if not e.get("meta_note"):
            for k in ("d_kind", "spec_log_weight"):  # a variant the catalog doesn't declare would be misdescribed
                if k not in e["meta"] and args.get(k) not in (None, "complex", 1.0):
                    raise AssertionError(f"{run}: undeclared {k}={args[k]}")
        return
    cfg = json.loads((RES / "e4" / run / "config.json").read_text())
    assert cfg.get("loss", "mss") == e.get("loss", "mss"), f"{run}: loss {cfg.get('loss')}"
    if p:
        a, b = overrides(p), cfg["overrides"]
        got = {k: (MINUS if k not in b else b[k]) for k in set(a) | set(b) if a.get(k, "·") != b.get(k, "·")}
        # noise_gate=False spelled out equals the default; compare what the recipe sets
        want = {k: v for k, v in e["diff"].items()}
        assert got == want, f"{run}: config differs from parent by {got}, catalog says {want}"


def main():
    metrics = json.loads((RES / "catalog_metrics.json").read_text())
    runs = {e["run"]: file_key(e) for e in C if e.get("run")}
    models = {}
    for e in C:
        run = e.get("run")
        finished = run is None or (RES / ("" if run.startswith("e6/") else "e4") / run / "model.pt").exists()
        m = e.get("fixed") or metrics.get(run)
        check(e, finished and m is not None)
        models[file_key(e)] = dict(
            name=e["n"], parent=runs.get(e.get("p")), rel=e.get("rel"), change=e["change"],
            loss=m["mean"] if m else None, wind=m["wind"] if m else None,
            **({"fav": True} if e.get("fav") else {}), **({"live": True} if e.get("live") else {}),
            **({"findings": True} if e.get("fixed") else {}), **({"pending": True} if m is None else {}))
    s = PAGE.read_text()
    a, b = "/*MODELS-START*/", "/*MODELS-END*/"
    i, j = s.index(a), s.index(b)
    PAGE.write_text(s[:i + len(a)] + "\nconst MODELS = " + json.dumps(models, ensure_ascii=False, indent=0) + ";\n" + s[j:])
    print(len(models), "models;", sum(1 for m in models.values() if m.get("pending")), "pending")


if __name__ == "__main__":
    main()
