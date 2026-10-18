"""Gradio app for the Garliclo ReAct agent: pick a persona, tune the prompt and configuration, watch the trace."""
import html
import json

import gradio as gr

import garliclo_react as agent

ASSETS_DIR = agent.BASE_DIR / "assets"
LOGO = ASSETS_DIR / "logo.png"
LABEL_TO_KEY = {p["label"]: k for k, p in agent.PERSONAS.items()}
MODELS = ["gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini"]
EXAMPLES = [
    "Hi! What kinds of garlic do you sell and how much per kilo?",
    "How much would 30 kg of fresh trimmed garlic delivered to Tripoli cost?",
    "We use 50 kg of frozen trimmed garlic a month in Dekwaneh. Monthly cost delivered, and how do we store it?",
    "Our last delivery had soft cloves and we're not happy.",
    "Do you deliver on Sundays?",
    "قديش سعر كيلو المقشر غير المقمع؟",
]


def esc(value) -> str:
    return html.escape(str(value))


def message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text") or ""
    if isinstance(content, (list, tuple)):
        return " ".join(message_text(c) for c in content)
    return ""


def clean_history(history: list) -> list[dict]:
    out = []
    for m in history or []:
        role, text = m.get("role"), message_text(m.get("content")).strip()
        if role in ("user", "assistant") and text:
            out.append({"role": role, "content": text})
    return out


def persona_card(label: str) -> str:
    p = agent.PERSONAS[LABEL_TO_KEY[label]]
    traits = "".join(f"<li>{esc(line.lstrip('- '))}</li>" for line in p["prompt"].splitlines()[1:])
    rows = "".join(f"<tr><td><code>{esc(k)}</code></td><td><b>{esc(v)}</b></td><td>{esc(p['why'].get(k, ''))}</td></tr>"
                   for k, v in p["config"].items())
    return (f"<div class='card'><div class='card-h'>{esc(p['label'])}</div><div class='muted'>Task: {esc(p['task'])}"
            f"</div><ul class='traits'>{traits}</ul><table class='cfg'>{rows}</table></div>")


def render_trace(result: dict | None) -> str:
    if not result:
        return ("<div class='empty'><div class='big'>🧠</div>Ask something to watch the ReAct loop: "
                "every Thought, Action and Observation appears here.</div>")
    parts = []
    for i, step in enumerate(result["steps"], 1):
        block = [f"<div class='step-n'>Step {i}</div>"]
        if step.get("thought"):
            block.append(f"<div class='row thought'>💭 <b>Thought</b> {esc(step['thought'])}</div>")
        if step.get("action"):
            args = json.dumps(step["action"]["args"], ensure_ascii=False)
            block.append(f"<div class='row action'>🛠️ <b>Action</b> <code>{esc(step['action']['tool'])}"
                         f"({esc(args)})</code></div>")
        if step.get("error"):
            block.append(f"<div class='row error'>⚠️ {esc(step['error'])}</div>")
        elif step.get("observation"):
            block.append(f"<div class='row obs'>👀 <b>Observation</b> {esc(step['observation'][:450])}</div>")
        if step.get("answer"):
            block.append("<div class='row answer'>✅ <b>Answer</b> (sent to the chat)</div>")
        parts.append(f"<div class='step'>{''.join(block)}</div>")
    c = result["config"]
    parts.append(f"<div class='stats'>{result['llm_calls']} LLM calls · {len(result['tools_used'])} tool calls · "
                 f"{result['tokens_in'] + result['tokens_out']:,} tokens · ${result['cost']:.4f} · "
                 f"{result['ms'] / 1000:.1f} s<br>{esc(c['model'])} · temp {c['temperature']} · top_p {c['top_p']} · "
                 f"freq {c['frequency_penalty']} · pres {c['presence_penalty']} · max {c['max_tokens']}</div>")
    return "".join(parts)


def respond(message, history, persona_label, cot, shots, placement, instructions, model, temperature, top_p,
            frequency_penalty, presence_penalty, max_tokens):
    variant = {"cot": cot, "shots": shots, "placement": placement, "instructions": instructions}
    overrides = {"model": model, "temperature": temperature, "top_p": top_p, "frequency_penalty": frequency_penalty,
                 "presence_penalty": presence_penalty, "max_tokens": int(max_tokens)}
    try:
        result = agent.run_react(message, LABEL_TO_KEY[persona_label], history=clean_history(history),
                                 variant=variant, config_overrides=overrides, verbose=False)
        return result["answer"], render_trace(result)
    except Exception as exc:
        return "Sorry, something went wrong. Please try again.", f"<div class='empty'>⚠️ {esc(exc)}</div>"


def persona_defaults(label: str):
    c = agent.PERSONAS[LABEL_TO_KEY[label]]["config"]
    return (persona_card(label), c["model"], c["temperature"], c["top_p"], c["frequency_penalty"],
            c["presence_penalty"], c["max_tokens"])


CSS = """
:root { --plum:#3A2733; --olive:#3E422C; --cream:#F4F1EA; --sand:#E4DFD3; --ink:#2B2228; }
.gradio-container { max-width: 1260px !important; margin: auto !important;
  background: radial-gradient(circle at 10% 0%, #FBF9F4 0%, #ECE8DF 55%, #E2DDD1 100%) !important; }
footer { display:none !important; }
#hero { display:flex; align-items:center; gap:16px; padding:12px 20px; border-radius:20px; background:var(--cream);
  border:1px solid var(--sand); box-shadow:0 8px 30px rgba(58,39,51,.08); flex-wrap:wrap; }
#hero img { width:78px; }
#hero h1 { font-family:'Fraunces', Georgia, serif; font-size:30px; margin:0; color:var(--olive); }
#hero p { margin:2px 0 0; color:var(--plum); font-size:14.5px; }
#hero .tag { margin-left:auto; background:#fff; border:1px solid var(--sand); padding:6px 14px; border-radius:999px;
  font-size:13px; color:var(--ink); }
#left, #right { background:rgba(255,255,255,.55); border:1px solid var(--sand); border-radius:20px; padding:12px !important; }
#persona .wrap { gap:8px !important; }
#persona label { border-radius:14px !important; padding:10px 14px !important; border:1px solid var(--sand) !important; }
#persona label.selected { background:var(--plum) !important; color:#fff !important; }
#chatbot .message p { unicode-bidi: plaintext; text-align: start; }
#chatbot .user { background:var(--plum) !important; color:#fff !important; border:none !important; }
#chatbot .user * { color:#fff !important; }
.empty { text-align:center; padding:28px 10px; color:#6b5f66; } .empty .big { font-size:34px; }
.step { background:#fff; border:1px solid var(--sand); border-radius:14px; padding:8px 11px; margin-bottom:8px; }
.step-n { font-size:11.5px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--plum); }
.row { font-size:13px; margin-top:4px; padding:5px 8px; border-radius:8px; unicode-bidi:plaintext; }
.thought { background:#F6F1F8; } .action { background:#FFF5E6; } .obs { background:#EEF6F0; }
.answer { background:#E8F0FB; } .error { background:#F8E1E1; }
.row code { font-size:12px; white-space:pre-wrap; word-break:break-word; }
.stats { font-size:12.5px; color:#6b5f66; text-align:right; margin-top:6px; }
.card { background:#fff; border:1px solid var(--sand); border-radius:14px; padding:12px 14px; }
.card-h { font-weight:800; font-size:16px; color:var(--plum); } .muted { color:#6b5f66; font-size:13px; }
.traits { font-size:13px; padding-left:18px; margin:8px 0; }
.cfg { width:100%; font-size:12.5px; border-collapse:collapse; } .cfg td { padding:4px 6px; border-top:1px solid var(--sand); }
@media (max-width: 720px) { #main-row { flex-direction: column !important; } #left, #right { width:100% !important; } }
"""
THEME = gr.themes.Soft(primary_hue="stone", neutral_hue="stone",
                       font=[gr.themes.GoogleFont("Cairo"), "ui-sans-serif", "sans-serif"]).set(
    body_background_fill="#ECE8DF", button_primary_background_fill="#3A2733",
    button_primary_background_fill_hover="#5E3A4D", button_primary_text_color="#FFFFFF")
HEAD = ("<link href='https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&display=swap' rel='stylesheet'>"
        "<style>@media (max-width:720px){.gradio-container .main{padding-left:8px!important;"
        "padding-right:8px!important}}</style>")
FORCE_LIGHT = "() => { document.body.classList.remove('dark'); }"
LAUNCH_KWARGS = {"theme": THEME, "css": CSS, "head": HEAD, "js": FORCE_LIGHT, "allowed_paths": [str(ASSETS_DIR)]}
gr.set_static_paths(paths=[ASSETS_DIR])

first = next(iter(LABEL_TO_KEY))
c0 = agent.PERSONAS[LABEL_TO_KEY[first]]["config"]

with gr.Blocks(title="Garliclo · ReAct personas") as demo:
    gr.HTML(f"""<div id='hero'><img src='/gradio_api/file={LOGO.as_posix()}' alt='Garliclo logo'>
        <div><h1>GARLICLO</h1><p>ReAct agent with configurable personas · LangGraph</p></div>
        <div class='tag'>💭 Thought → 🛠️ Action → 👀 Observation → ✅ Answer</div></div>""")
    with gr.Row(elem_id="main-row"):
        with gr.Column(scale=3, elem_id="left"):
            persona = gr.Radio(list(LABEL_TO_KEY), value=first, label="Persona", elem_id="persona")
            cot = gr.Checkbox(True, label="Chain-of-Thought (Thought lines)", render=False)
            shots = gr.Radio(["few", "zero"], value="few", label="Examples (few-shot / zero-shot)", render=False)
            placement = gr.Radio(["system", "in_message"], value="system", label="Instructions placement", render=False)
            instructions = gr.Checkbox(True, label="Include the instructions block", render=False)
            model = gr.Dropdown(MODELS, value=c0["model"], label="Model", render=False)
            temperature = gr.Slider(0, 1.5, value=c0["temperature"], step=0.05, label="temperature", render=False)
            top_p = gr.Slider(0.05, 1, value=c0["top_p"], step=0.05, label="top_p", render=False)
            freq = gr.Slider(0, 2, value=c0["frequency_penalty"], step=0.1, label="frequency_penalty", render=False)
            pres = gr.Slider(0, 2, value=c0["presence_penalty"], step=0.1, label="presence_penalty", render=False)
            max_tokens = gr.Slider(60, 800, value=c0["max_tokens"], step=10, label="max_tokens", render=False)
            trace = gr.HTML(render_trace(None), render=False)
            gr.ChatInterface(
                fn=respond,
                chatbot=gr.Chatbot(elem_id="chatbot", height=500, show_label=False, avatar_images=(None, str(LOGO)),
                                   placeholder="<div class='empty'><div class='big'>🧄</div><b>Pick a persona and ask "
                                               "about garlic, prices, delivery or a problem.</b></div>"),
                textbox=gr.Textbox(placeholder="Ask the Garliclo agent… اسألني أي شي", show_label=False,
                                   submit_btn="Send"),
                additional_inputs=[persona, cot, shots, placement, instructions, model, temperature, top_p, freq,
                                   pres, max_tokens],
                additional_inputs_accordion=gr.Accordion("⚙️ Prompt & LLM configuration", open=False, render=False),
                additional_outputs=[trace],
                examples=[[e] for e in EXAMPLES], examples_label="Try asking", cache_examples=False,
                fill_height=False,
            )
        with gr.Column(scale=2, elem_id="right"):
            with gr.Tabs():
                with gr.Tab("🧠 ReAct trace"):
                    trace.render()
                with gr.Tab("🎭 Persona"):
                    card = gr.HTML(persona_card(first))
    persona.change(persona_defaults, persona, [card, model, temperature, top_p, freq, pres, max_tokens])

if __name__ == "__main__":
    demo.launch(**LAUNCH_KWARGS)
