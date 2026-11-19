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
    # experiment 10 (2026-11-16): attack vs noise, network backbones, strings from the tablature
    dict(n="M2 + timbre 32 + ataque por nota", run="structured_v10_m2_burst", p="structured_v9_t32_no_gate", rel="rec", diff=dict(rich_burst=True), change="La ráfaga de ataque deja de ser un solo espectro por grabación con un decaimiento global: cada nota tiene su propio nivel y su propio decaimiento en cada banda, según su altura, volumen, timbre y la grabación."),
    dict(n="M1 + timbre 32 + ataque por nota", run="structured_v10_m1_burst", p="structured_v9_base_t32", rel="rec", diff=dict(rich_burst=True), change="El mismo ataque por nota, sobre M1 + timbre 32, donde el ruido hace de «tambor»."),
    dict(n="M2 + timbre 32 + ruido lento", run="structured_v10_m2_slow", p="structured_v9_t32_no_gate", rel="rec", diff=dict(slow_noise=0.2), change="El ruido residual solo puede cambiar lento: sus bandas se suavizan en 200 ms, así que no puede seguir cada ataque ni hacer ráfagas."),
    dict(n="M2 + timbre 32 + fondo medido", run="structured_v10_m2_floor", p="structured_v9_t32_no_gate", rel="rec", diff=dict(room_floor=True), change="El ruido residual es gratis hasta el fondo medido de cada grabación (lo más bajo que se oye entre los parciales mientras suena la pieza) y se penaliza por encima."),
    dict(n="M1 + timbre 32 + ataque por nota + fondo medido", run="structured_v10_m1_burst_floor", p="structured_v10_m1_burst", rel="rec", diff=dict(room_floor=True), change="Ataque por nota y fondo medido, sobre M1."),
    dict(n="M2 + timbre 32 · LSTM", run="structured_v10_m2_lstm", p="structured_v9_t32_no_gate", rel="rec", diff=dict(backbone="lstm"), change="La red recurrente es una LSTM en vez de una GRU (como en Jonason), del mismo tamaño."),
    dict(n="M2 + timbre 32 · LSTM bidireccional", run="structured_v10_m2_bilstm", p="structured_v9_t32_no_gate", rel="rec", diff=dict(backbone="bilstm"), change="Una LSTM que también lee hacia atrás: cada cuadro ve lo que viene. Mismo tamaño total; no sirve en tiempo real."),
    dict(n="M2 + timbre 32 · atención", run="structured_v10_m2_attn", p="structured_v9_t32_no_gate", rel="rec", diff=dict(backbone="attn"), change="En vez de la GRU, dos capas de self-attention: cada cuadro de cada voz mira los últimos 64 cuadros (640 ms) de esa misma voz."),
    dict(n="M2 + timbre 32 + cuerdas", run="structured_v10_m2_strings", p="structured_v9_t32_no_gate", rel="rec", diff=dict(strings=True), change="Cada voz es una cuerda, según la tablatura alineada al MIDI (99 % de las notas). La red sabe qué cuerda y qué traste; la inarmonicidad es la medida en cada cuerda y sube con el traste (las entorchadas resultaron 3–4 veces menos rígidas que lo que decía la ley única)."),
    dict(n="M2 + timbre 32 + cuerdas · atención entre cuerdas", run="structured_v10_m2_strings_xvoice", p="structured_v10_m2_strings", rel="rec", diff=dict(cross_voice=True), change="Además, una capa de self-attention entre las 6 cuerdas en cada cuadro."),
    dict(n="M2 + timbre 32 + adv · contacto", run="e6/structured_v9_t32_no_gate_noise_strongD_contact_rise", p="structured_v9_t32_no_gate", rel="cont", meta=dict(overrides=dict(contact=True, delta_rise=20.0), d_kind="complex", **ADV), change="Como M2 + timbre 32 + adv, con un ruido de contacto por voz (dedo o uña tocando la cuerda) y una penalización a las subidas rápidas de los armónicos agudos: así imitaba el adversarial ese contacto, con los «agudos cortos»."),
    dict(n="M1 + timbre 32 + ataque por nota + adv", run="e6/structured_v10_m1_burst_noise_strongD", p="structured_v10_m1_burst", rel="cont", meta=dict(overrides={}, d_kind="complex", **ADV), change="La receta del favorito (adversarial de 3000 pasos) sobre M1 + timbre 32 + ataque por nota: el favorito más el ataque por nota."),
    dict(n="M2 + timbre 32 + cuerdas + adv", run="e6/structured_v10_m2_strings_noise_strongD", p="structured_v10_m2_strings", rel="cont", meta=dict(overrides={}, d_kind="complex", **ADV), change="Adversarial de 3000 pasos sobre M2 + timbre 32 + cuerdas."),
    dict(n="M2 + timbre 32 + cuerdas · atención", run="structured_v10_m2_strings_attn", p="structured_v10_m2_strings", rel="rec", diff=dict(backbone="attn"), change="Cuerdas, y en vez de la GRU la self-attention sobre los últimos 64 cuadros de cada cuerda."),
    dict(n="M1 + timbre 32 + ataque por nota + cuerdas", run="structured_v10_m1_burst_strings", p="structured_v10_m1_burst", rel="rec", diff=dict(strings=True), change="El ataque por nota (que le quitó el «tambor» al ruido de M1) más las cuerdas."),
    dict(n="M1 + timbre 32 + ataque por nota + cuerdas + adv", run="e6/structured_v10_m1_burst_strings_noise_strongD", p="structured_v10_m1_burst_strings", rel="cont", meta=dict(overrides={}, d_kind="complex", **ADV), change="La receta del favorito (adversarial de 3000 pasos) sobre M1 + timbre 32 + ataque por nota + cuerdas."),
    dict(n="M2 + timbre 32 + cuerdas + adv · sin agudos", run="e6/structured_v10_m2_strings_noise_strongD_rise", p="structured_v10_m2_strings", rel="cont", meta=dict(overrides=dict(delta_rise=20.0), d_kind="complex", **ADV), change="Como M2 + timbre 32 + cuerdas + adv, con la penalización a las subidas rápidas de los armónicos agudos que quitó los «agudos cortos»."),
    dict(n="M2 + timbre 32 + cuerdas · semilla 1", run="structured_v10_m2_strings_seed1", p="structured_v10_m2_strings", rel="rec", diff={}, change="Lo mismo que M2 + timbre 32 + cuerdas con otra semilla: ¿la mejora de las cuerdas se repite?"),
    dict(n="M1 + timbre 32 + ataque por nota + fondo medido + cuerdas", run="structured_v10_m1_burst_floor_strings", p="structured_v10_m1_burst_floor", rel="rec", diff=dict(strings=True), change="El M1 sin tambor y con menos viento, más las cuerdas."),
    dict(n="M2 + timbre 32 + cuerdas · LSTM bidireccional", run="structured_v10_m2_strings_bilstm", p="structured_v10_m2_strings", rel="rec", diff=dict(backbone="bilstm"), change="Cuerdas con la LSTM bidireccional en vez de la GRU (no sirve en tiempo real)."),
    dict(n="M1 + timbre 32 + ataque por nota + fondo medido + cuerdas + adv", run="e6/structured_v10_m1_burst_floor_strings_noise_strongD", p="structured_v10_m1_burst_floor_strings", rel="cont", meta=dict(overrides={}, d_kind="complex", **ADV), change="La receta del favorito sobre el M1 sin tambor, con menos viento y con cuerdas."),
    # experiment 11: the final model (all on M1 + timbre 32 + per-note attack + room background, no strings)
    dict(n="M1 + timbre 32 + ataque por nota + fondo medido + adv · sin agudos", run="e6/structured_v10_m1_burst_floor_noise_strongD_rise", p="structured_v10_m1_burst_floor", rel="cont", meta=dict(overrides=dict(delta_rise=20.0), d_kind="complex", **ADV), change="Control de los candidatos finales: la receta del favorito, con la penalización de los agudos cortos, sobre M1 con ataque por nota y fondo medido (GRU)."),
    dict(n="M1 + timbre 32 + ataque por nota + fondo medido · atención", run="structured_v11_m1_attn", p="structured_v10_m1_burst_floor", rel="rec", diff=dict(backbone="attn"), change="La atención con ventana (la que preferiste en M2) en vez de la GRU, ahora sobre M1."),
    dict(n="M1 + timbre 32 + ataque por nota + fondo medido · atención + adv · sin agudos", run="e6/structured_v11_m1_attn_noise_strongD_rise", p="structured_v11_m1_attn", rel="cont", meta=dict(overrides=dict(delta_rise=20.0), d_kind="complex", **ADV), change="Candidato final: la receta del favorito sobre M1 con atención."),
    dict(n="M1 + timbre 32 + ataque por nota + fondo medido · atención que anticipa", run="structured_v11_m1_attn_ahead", p="structured_v11_m1_attn", rel="rec", diff=dict(attn_ahead=16), change="La atención también ve 160 ms hacia adelante en la partitura (como la LSTM bidireccional, que ganó en los rasgueos). Solo sirve fuera de tiempo real."),
    dict(n="M1 + timbre 32 + ataque por nota + fondo medido · atención que anticipa + adv · sin agudos", run="e6/structured_v11_m1_attn_ahead_noise_strongD_rise", p="structured_v11_m1_attn_ahead", rel="cont", meta=dict(overrides=dict(delta_rise=20.0), d_kind="complex", **ADV), change="Candidato final: la receta del favorito sobre M1 con atención que anticipa."),
    dict(n="M1 + timbre 32 + ataque por nota + fondo medido · atención que anticipa + entre voces", run="structured_v11_m1_attn_ahead_xvoice", p="structured_v11_m1_attn_ahead", rel="rec", diff=dict(cross_voice=True), change="Además, en cada instante cada voz ve lo que hacen las otras (para los rasgueos)."),
    dict(n="M1 + timbre 32 + ataque por nota + fondo medido · atención que anticipa + entre voces + adv · sin agudos", run="e6/structured_v11_m1_attn_ahead_xvoice_noise_strongD_rise", p="structured_v11_m1_attn_ahead_xvoice", rel="cont", meta=dict(overrides=dict(delta_rise=20.0), d_kind="complex", **ADV), change="Candidato final: la receta del favorito sobre M1 con atención que anticipa y entre voces."),
    dict(n="M1 + timbre 32 + ataque por nota + fondo medido · atención que anticipa + contacto", run="structured_v11_m1_attn_ahead_contact", p="structured_v11_m1_attn_ahead", rel="rec", diff=dict(contact=True, contact_init=-8.0, delta_rise=20.0), change="Una parte propia para los eventos agudos y rápidos (dedo o uña tocando la cuerda), entrenada desde cero, con la penalización de los agudos cortos también en el entrenamiento: los armónicos ya no pueden hacerlos."),
    dict(n="M1 + timbre 32 + ataque por nota + fondo medido · atención que anticipa + contacto + adv · sin agudos", run="e6/structured_v11_m1_attn_ahead_contact_noise_strongD_rise", p="structured_v11_m1_attn_ahead_contact", rel="cont", meta=dict(overrides=dict(delta_rise=20.0), d_kind="complex", **ADV), change="Candidato final: la receta del favorito sobre M1 con atención que anticipa y la parte de contacto."),
    dict(n="M1 + timbre 32 + ataque por nota + fondo medido · atención que anticipa + grabación en todo", run="structured_v11_m1_attn_ahead_rec", p="structured_v11_m1_attn_ahead", rel="rec", diff=dict(rec_all=True), change="Todas las partes que pueden depender de la guitarra, el micrófono o la sala leen la grabación: el decaimiento de cada parcial, un EQ fino del cuerpo (24 bandas hasta 800 Hz, puede mover las resonancias de cada guitarra), el tiempo de ataque y el apagado al final de la nota, y la rigidez (cada guitarra medida respecto al promedio)."),
    dict(n="M1 + timbre 32 + ataque por nota + fondo medido · atención que anticipa + grabación en todo + adv · sin agudos", run="e6/structured_v11_m1_attn_ahead_rec_noise_strongD_rise", p="structured_v11_m1_attn_ahead_rec", rel="cont", meta=dict(overrides=dict(delta_rise=20.0), d_kind="complex", **ADV), change="Candidato final: la receta del favorito sobre M1 con atención que anticipa y la grabación en todas las partes."),
    dict(n="M2 + timbre 32 + ataque por nota + grabación en todo", run="structured_v11_m2_rec", p="structured_v10_m2_burst", rel="rec", diff=dict(rec_all=True), change="M2 con todas sus partes leyendo la grabación: además de lo de «grabación en todo» en M1 (decaimiento, EQ fino del cuerpo, ahora hasta 8 kHz, ataque, apagado y rigidez), el doble decaimiento y el batido, y el espectro del clic. ¿M2 falló porque esas partes eran iguales para todas las guitarras?"),
    dict(n="M1 + timbre 32 + ataque por nota + fondo medido · atención que anticipa + grabación en todo · embedding 32", run="structured_v11_m1_attn_ahead_rec32", p="structured_v11_m1_attn_ahead_rec", rel="rec", diff=dict(rec_dim=32), change="El mismo modelo con un embedding de grabación de 32 dimensiones en vez de 16: con 20 grabaciones y 16 dimensiones, las lecturas lineales por grabación no pueden darle a cada una un valor independiente."),
    # experiment 11: why strings sound synthetic (from M1 + per-note attack + room background + strings, GRU)
    dict(n="M1 + timbre 32 + ataque por nota + fondo medido + cuerdas · rigidez antigua", run="structured_v11_m1_strings_oldB", p="structured_v10_m1_burst_floor_strings", rel="rec", diff=dict(strings_b="pitch"), change="Cuerdas, pero con la ley de rigidez de antes (una recta sobre la altura): los bordones vuelven a ser más inarmónicos."),
    dict(n="M1 + timbre 32 + ataque por nota + fondo medido + cuerdas · rigidez por guitarra", run="structured_v11_m1_strings_guitarB", p="structured_v10_m1_burst_floor_strings", rel="rec", diff=dict(strings_b="guitar"), change="La rigidez de cada cuerda medida en cada grabación (cada una es otra guitarra y otro juego de cuerdas), no un promedio de todas."),
    dict(n="M1 + timbre 32 + ataque por nota + fondo medido + cuerdas · decaimiento por cuerda", run="structured_v11_m1_strings_decay", p="structured_v10_m1_burst_floor_strings", rel="rec", diff=dict(decay_cond=True), change="El decaimiento de cada parcial también sabe la cuerda, el traste y la grabación (antes solo la altura y la frecuencia)."),
    dict(n="M1 + timbre 32 + ataque por nota + fondo medido + cuerdas · voces de antes", run="structured_v11_m1_strings_oldvoices", p="structured_v10_m1_burst_floor_strings", rel="rec", diff=dict(strings_voices=False), change="La red sabe la cuerda y el traste, pero las notas se reparten en voces como antes, no una voz por cuerda."),
    # experiment 12: the final DDSP (T4 + strings + the recording in every part, 32), then the neural vocoder
    dict(n="Final · rigidez compartida", run="structured_v12_final", p="structured_v11_m1_attn_ahead_contact", rel="rec", diff=dict(strings=True, rec_all=True, rec_dim=32, rec_b="none"), change="El modelo final: T4 (atención que anticipa + contacto) + cuerdas de la tablatura + la grabación en todas las partes (embedding de 32). Una tabla de rigidez por cuerda, igual para todas las guitarras."),
    dict(n="Final · rigidez aprendida", run="structured_v12_final_learnedB", p="structured_v11_m1_attn_ahead_contact", rel="rec", diff=dict(strings=True, rec_all=True, rec_dim=32, rec_b="learned"), change="Igual, pero la rigidez de cada cuerda se corrige por guitarra desde el embedding de la grabación (aprendida, no medida)."),
    dict(n="Final · rigidez compartida + adv", run="e6/structured_v12_final_noise_strongD_rise", p="structured_v12_final", rel="cont", meta=dict(overrides=dict(delta_rise=20.0), d_kind="complex", **ADV), change="La receta adversarial del favorito (3000 pasos, sin agudos)."),
    dict(n="Final · rigidez aprendida + adv", run="e6/structured_v12_final_learnedB_noise_strongD_rise", p="structured_v12_final_learnedB", rel="cont", meta=dict(overrides=dict(delta_rise=20.0), d_kind="complex", **ADV), change="La receta adversarial del favorito (3000 pasos, sin agudos). El DDSP final de la tesis."),
]
# Vocoder versions (experiment 12): the same audio passed through a neural vocoder. key: file name; src: what went in
# (a catalog key, "original", or None for a reference); loss on the 85 held-out windows from e12_voc_metrics.json.
VOC = [
    dict(n="Original → BigVGAN 24 kHz", key="voc-bigvgan24-original", src="original", change="La grabación real convertida a mel y de vuelta a audio por BigVGAN-v2 preentrenado (24 kHz, sin entrenar en guitarra): el techo de un vocoder de mel."),
    dict(n="Final + adv → BigVGAN 24 kHz", key="voc-bigvgan24-e6-structured_v12_final_learnedB_noise_strongD_rise", src="e6-structured_v12_final_learnedB_noise_strongD_rise", change="El DDSP final por BigVGAN-v2 preentrenado, sin entrenar en guitarra."),
    dict(n="Original 44.1 kHz", key="original44k", src=None, change="La grabación real en banda completa (44.1 kHz), sin pasar por ningún modelo."),
    dict(n="Original → BigVGAN 44 kHz", key="voc-bigvgan44-original48", src="original44k", change="La grabación real en banda completa por BigVGAN-v2 preentrenado de 44 kHz."),
    dict(n="Original → DisCoder", key="voc-discoder-original44k", src="original44k", change="La grabación real en banda completa por DisCoder (vocoder de música, 44.1 kHz, preentrenado)."),
    dict(n="Final + adv → BigVGAN 44 kHz", key="voc-bigvgan44-e6-structured_v12_final_learnedB_noise_strongD_rise", src="e6-structured_v12_final_learnedB_noise_strongD_rise", change="El DDSP final (24 kHz) por BigVGAN-v2 de 44 kHz: tiene que inventar lo que está por encima de 12 kHz."),
    dict(n="Final + adv → DisCoder", key="voc-discoder-e6-structured_v12_final_learnedB_noise_strongD_rise", src="e6-structured_v12_final_learnedB_noise_strongD_rise", change="El DDSP final (24 kHz) por DisCoder: también inventa lo que está por encima de 12 kHz."),
    dict(n="Original → BigVGAN afinado", key="voc-p2_real-original", src="original", change="La grabación real por BigVGAN-v2 de 24 kHz después de afinarlo 8000 pasos con las grabaciones reales de GAPS (fase 2): el nuevo techo."),
    dict(n="Final + adv → BigVGAN afinado", key="voc-p2_real-e6-structured_v12_final_learnedB_noise_strongD_rise", src="e6-structured_v12_final_learnedB_noise_strongD_rise", change="El DDSP final por el BigVGAN afinado en guitarra real. Nunca vio un DDSP en el entrenamiento."),
    dict(n="Final + adv → mel reparado (difusión) → BigVGAN afinado", key="voc-p2_real-v3_x0res-d20-e6-structured_v12_final_learnedB_noise_strongD_rise", src="e6-structured_v12_final_learnedB_noise_strongD_rise", change="El mel del DDSP pasa por una red de difusión superficial (se le añade ruido hasta el paso 20 de 100 y se quita, aprendiendo a dejarlo como el mel real) y luego por el BigVGAN afinado."),
    dict(n="Final + adv → mel reparado (flujo) → BigVGAN afinado", key="voc-p2_real-v3b_flow-e6-structured_v12_final_learnedB_noise_strongD_rise", src="e6-structured_v12_final_learnedB_noise_strongD_rise", change="El mel del DDSP pasa por un flujo rectificado que lo lleva en línea recta hacia el mel real, y luego por el BigVGAN afinado."),
    dict(n="Final + adv → vocoder entrenado con DDSP (V1)", key="voc-v1_mix20-e6-structured_v12_final_learnedB_noise_strongD_rise", src="e6-structured_v12_final_learnedB_noise_strongD_rise", change="El BigVGAN afinado, entrenado 6000 pasos más con pares DDSP → grabación real: entra el mel del DDSP final y tiene que salir la grabación real (pérdida de mel + adversarial). 20 grabaciones, 0.69 h de pares."),
    dict(n="Final + adv → vocoder con la onda del DDSP (V2)", key="voc-v2_inject20-e6-structured_v12_final_learnedB_noise_strongD_rise", src="e6-structured_v12_final_learnedB_noise_strongD_rise", change="Como V1, pero además la onda del DDSP entra en cada etapa del vocoder (como la fuente en NSF-HiFiGAN), no solo su mel: el vocoder puede copiar la fase y los ataques del DDSP en vez de inventarlos. 20 grabaciones, 0.69 h de pares."),
    dict(n="Final + adv → vocoder con cada parte del DDSP (S)", key="voc-s_stems20b-e6-structured_v12_final_learnedB_noise_strongD_rise", src="e6-structured_v12_final_learnedB_noise_strongD_rise", change="Como V1, pero además del mel de la mezcla entra el mel de cada parte del DDSP por separado (cuerdas, cuerpo, ataque, contacto, fondo, ruido), para que el vocoder sepa qué es cada cosa. 20 grabaciones, 0.69 h de pares."),
    dict(n="Solo las cuerdas del DDSP → V2 (fase 4)", key="voc-p4_v2_strings20-e6-structured_v12_final_learnedB_noise_strongD_rise", src="stems/e6-structured_v12_final_learnedB_noise_strongD_rise/strings", change="V2 entrenado con solo la parte armónica del DDSP (las cuerdas): sin cuerpo, ataque, contacto, fondo ni ruido. Todo lo demás lo tiene que poner el vocoder. «Entrada» es la parte de cuerdas sola."),
    dict(n="Original 44.1 kHz → BigVGAN 44 kHz afinado", key="voc-p5_real44-original", src="original44k", change="La grabación real en banda completa por BigVGAN-v2 de 44 kHz después de afinarlo 8000 pasos con las grabaciones reales de GAPS a 44.1 kHz (fase 5, como la fase 2 a 24 kHz)."),
    dict(n="Final + adv → BigVGAN 44 kHz afinado", key="voc-p5_real44-e6-structured_v12_final_learnedB_noise_strongD_rise", src="e6-structured_v12_final_learnedB_noise_strongD_rise", change="El DDSP final (24 kHz) por el BigVGAN de 44 kHz afinado en guitarra real: nunca vio un DDSP y tiene que inventar lo que está por encima de 12 kHz."),
]


def kad_values():
    """{page key: [PANNs-KAD, EnCodec-KAD]} against the same 85 real windows (analysis/e12_kad.py)."""
    f = RES / "e12_kad.json"
    if not f.exists():
        return {}
    k = json.loads(f.read_text())
    p, e = k["panns-wavegram-logmel vs real"]["versions"], k["encodec-emb vs real"]["versions"]
    L = "e6-structured_v12_final_learnedB_noise_strongD_rise"
    out = {}
    for v in p:
        if v.startswith("model-"):
            key = v[6:]  # a DDSP model rendered window by window (as its loss)
        elif v.startswith("diff-"):
            key = {"diff-v3_x0res": f"voc-p2_real-v3_x0res-d20-{L}", "diff-v3b_flow": f"voc-p2_real-v3b_flow-{L}"}.get(v)
        elif v.startswith("voc-"):
            run = v[4:].removesuffix("-real")
            run = {"p0_pretrained": "bigvgan24", "pretrained44": "bigvgan44"}.get(run, run)
            real = "original48" if run == "bigvgan44" else "original"  # the 44 kHz page file was cut from the 48 kHz file
            key = f"voc-{run}-" + (real if v.endswith("-real") else L)
        else:
            key = None  # "ddsp": the vocoders' input (pair renders); the page's final DDSP uses its window render
        if key and v in e:
            out[key] = [p[v]["kad"], e[v]["kad"]]
    return out


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
    vm = json.loads((RES / "e12_voc_metrics.json").read_text()) if (RES / "e12_voc_metrics.json").exists() else {}
    for e in VOC:
        m = vm.get(e["key"])
        models[e["key"]] = dict(name=e["n"], parent=e["src"] if e["src"] in models else None,
                                rel="voc" if e["src"] else None, change=e["change"], loss=m["mean"] if m else None,
                                wind="na", **({"pending": True} if m is None and e["src"] else {}),
                                **({"ref": True} if not e["src"] else {}))
    for k, v in kad_values().items():
        if k in models:
            models[k]["kad"] = v
    s = PAGE.read_text()
    a, b = "/*MODELS-START*/", "/*MODELS-END*/"
    i, j = s.index(a), s.index(b)
    PAGE.write_text(s[:i + len(a)] + "\nconst MODELS = " + json.dumps(models, ensure_ascii=False, indent=0) + ";\n" + s[j:])
    print(len(models), "models;", sum(1 for m in models.values() if m.get("pending")), "pending")


if __name__ == "__main__":
    main()
