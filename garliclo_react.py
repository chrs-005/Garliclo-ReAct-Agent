# %% [markdown]
# # 🧄 Garliclo ReAct Agent with Configurable Personas
# **EECE 503P/798S: Agentic Systems — C4: Agent Personas & Thinking Mechanisms**
#
# This notebook turns the C3 Garliclo business chatbot into a **ReAct agent** built with **LangGraph**. The
# Thought → Action → Observation loop is written by hand as graph nodes and conditional edges; no prebuilt executor
# (no `AgentExecutor`, no `create_react_agent`) is used. Three personas, each defined by **prompt traits and a matching
# LLM configuration**, are compared across prompt-engineering variants and configuration sweeps.
#
# | Grading item | Points | Where |
# |---|---|---|
# | Use-case summary and write-up | 10 | §1 |
# | ReAct logic manually implemented | 30 | §3 tools, §7 prompt, §8 LangGraph loop, §9 demo |
# | Persona design (prompt and configuration) | 20 | §5 |
# | Configuration experimentation and comparison | 15 | §10–§12 |
# | Clear instructions design | 10 | §4 |
# | Creativity, clarity, and reflection quality | 15 | §13 reflection, Gradio app |

# %% [markdown]
# ## 0. Setup
# Keys are read from `.env` (`OPENAI_API_KEY`, and `GEMINI_API_KEY` for the top-k experiment). `.env` is never committed.

# %%
import truststore

truststore.inject_into_ssl()  # use the operating system's certificate store for HTTPS calls

import ast
import math
import datetime
import json
import os
import re
import time
from pathlib import Path
from typing import TypedDict

import httpx
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from pypdf import PdfReader

load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ME_DIR, LOG_DIR = BASE_DIR / "me", BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=2, timeout=60)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAX_LLM_CALLS = 6  # guardrail: at most 6 reasoning steps per question

print("✅ OpenAI key loaded" if os.getenv("OPENAI_API_KEY") else "⚠️ OPENAI_API_KEY missing")

# %% [markdown]
# ## 1. Use case
# **Garliclo | Nader Garlic** is a real Lebanese supplier of air-peeled garlic, founded in 1980 by Joseph Nader
# (main factory in Sin El Fil, delivery all over Lebanon). The scenario is reused from C3, as the assignment allows.
#
# | | |
# |---|---|
# | **Agent goal** | Help food businesses buy peeled garlic: answer product questions, give exact quotes with delivery, plan quantities and storage, capture sales leads, and log complaints or questions the team must answer. |
# | **Target audience** | Restaurant owners and chefs (shawarma and grill shops), industrial kitchens, butcheries, supermarkets and caterers in Lebanon, writing in English, Arabic or French. Households are not served. |
# | **Tools** | `search_business_info` (profile lookup), `get_delivery_fee` (zone table), `calculate_quote` (exact totals), `record_customer_interest` (save a lead), `record_feedback` (log feedback or unanswered questions). |
# | **Constraints** | Only facts from the business files; all arithmetic done by a tool, never by the model; no invented discounts, refunds or delivery dates; leads saved only with a real email; business customers only; at most 6 reasoning steps per question; replies under 120 words. |
# | **Limitations** | No live stock, order or payment system: the agent cannot confirm availability, place or cancel orders, or take payment, so it hands those to the team (via the logging tools). Delivery fees, pack sizes and shelf life partly come from assumptions in the business file. Search is keyword-based (BM25), so unusual wording can miss a passage. |
#
# The three personas split the work by **task**: a *Friendly Advisor* for first contact and sales, a *Strict Expert*
# for exact quotes and kitchen planning, and a *Cautious Helper* for after-sales support.

# %% [markdown]
# ## 2. Business knowledge
# The C3 business files (`me/business_summary.txt`, `me/about_business.pdf`) are loaded and split into passages for the
# search tool. Products and delivery zones are parsed into tables so the quote tool can compute exact totals.

# %%
business_summary = (ME_DIR / "business_summary.txt").read_text(encoding="utf-8")
profile_text = "\n".join(page.extract_text() or "" for page in PdfReader(ME_DIR / "about_business.pdf").pages)

PRODUCTS = {
    i + 1: {"name": en.strip(), "arabic": ar.strip(), "price": float(price)}
    for i, (en, ar, price) in enumerate(re.findall(
        r"^\d+\.[ \t]*(.+?)[ \t]+—[ \t]+(.+?)[ \t]+—[ \t]+\$(\d+\.\d{2}) per kg", business_summary, re.M))
}
ZONES = [
    {"zone": int(n), "name": name.strip(), "areas": [a.strip() for a in areas.split(",")], "fee": int(fee)}
    for n, name, areas, fee in re.findall(r"^Zone (\d) — (.+?): (.+?) — \$(\d+)[ \t]*$", business_summary, re.M)
]
FREE_DELIVERY_KG, FREE_DELIVERY_MAX_ZONE = 50, 3

# passages for the search tool: summary paragraphs + PDF sections
PASSAGES = [p.strip() for p in business_summary.split("\n\n") if p.strip()]
PASSAGES += [s.strip() for s in re.split(r"\n(?=\d{1,2}\. [A-Z])", profile_text) if len(s.strip()) > 40]

print(f"{len(PRODUCTS)} products, {len(ZONES)} delivery zones, {len(PASSAGES)} searchable passages")
for pid, p in PRODUCTS.items():
    print(f"  product {pid}: {p['name']} — ${p['price']:.2f}/kg")

# %% [markdown]
# ## 3. Tools
# Each tool is a plain Python function that returns a short text **observation**. The `TOOLS` registry holds the
# description and parameter list that are embedded in the system prompt, and that the action parser uses to map
# arguments. Arithmetic lives in `calculate_quote`, so the model never computes totals itself.

# %%
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
STOPWORDS = set("the a an and or of to in for is are do does you your we our what how much many can i it on at by with "
                "from this that be as about any who which where when have has please".split())


def _stem(word: str) -> str:
    """Light stemmer so 'days'/'day' and 'founded'/'founder' match."""
    for suffix in ("ing", "ed", "er", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def _words(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9\u0600-\u06ff]+", text.lower())]


def _tokens(text: str) -> set[str]:
    return {t for t in _words(text) if t not in STOPWORDS}


def _append_log(filename: str, entry: dict) -> None:
    with open(LOG_DIR / filename, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.datetime.now().isoformat(timespec="seconds"), **entry},
                           ensure_ascii=False) + "\n")


def find_zone(area: str) -> dict | None:
    a = area.strip().lower()
    for z in ZONES:
        names = [x.lower() for x in z["areas"]] + [z["name"].lower()]
        if any(a == n or a in n or n in a for n in names if len(n) > 2):
            return z
    return None


PASSAGE_TOKENS = [_words(p) for p in PASSAGES]
AVG_LEN = sum(map(len, PASSAGE_TOKENS)) / len(PASSAGE_TOKENS)


def bm25(query_terms: set[str], doc: list[str], k1: float = 1.5, b: float = 0.75) -> float:
    """BM25 relevance: rare words (e.g. 'storage') count more than common ones (e.g. 'garlic')."""
    n = len(PASSAGE_TOKENS)
    score = 0.0
    for term in query_terms:
        tf = doc.count(term)
        if tf:
            df = sum(term in d for d in PASSAGE_TOKENS)
            idf = max(0.1, math.log((n - df + 0.5) / (df + 0.5) + 1))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / AVG_LEN))
    return score


def search_business_info(query: str) -> str:
    """Return the two passages of the business profile that best match the query (BM25 ranking)."""
    q = _tokens(query)
    scored = sorted(zip((bm25(q, d) for d in PASSAGE_TOKENS), PASSAGES), key=lambda x: x[0], reverse=True)
    hits = [p for s, p in scored[:2] if s > 0]
    return "\n---\n".join(h[:700] for h in hits) if hits else "No matching information in the business profile."


def get_delivery_fee(area: str) -> str:
    """Delivery zone and fee for an area in Lebanon."""
    z = find_zone(area)
    if not z:
        return f"'{area}' is not in the delivery table; the Garliclo team confirms the fee for unlisted areas."
    return (f"{area}: Zone {z['zone']} ({z['name']}), delivery fee ${z['fee']} per order. Orders of "
            f"{FREE_DELIVERY_KG} kg or more get free delivery in Zones 1-{FREE_DELIVERY_MAX_ZONE}.")


def calculate_quote(product_id: int, kg: float, area: str = "") -> str:
    """Exact order total: price x kg, plus the delivery fee for the area (with the free-delivery rule)."""
    try:
        product, kg = PRODUCTS[int(product_id)], float(kg)
    except (KeyError, ValueError, TypeError):
        return f"Error: product_id must be one of {list(PRODUCTS)} and kg a number."
    if kg <= 0:
        return "Error: kg must be positive."
    subtotal = round(product["price"] * kg, 2)
    line = f"Product {int(product_id)} ({product['name']}): {kg:g} kg x ${product['price']:.2f} = ${subtotal:.2f}"
    z = find_zone(area) if area else None
    if not z:
        return f"{line} | Delivery: {'no area given' if not area else area + ' not in the table'} (team confirms) | " \
               f"Total before delivery: ${subtotal:.2f}"
    free = kg >= FREE_DELIVERY_KG and z["zone"] <= FREE_DELIVERY_MAX_ZONE
    fee = 0 if free else z["fee"]
    delivery = f"FREE (Zone {z['zone']}, {FREE_DELIVERY_KG} kg+)" if free else f"Zone {z['zone']}, ${fee}"
    return f"{line} | Delivery to {area}: {delivery} | Total: ${subtotal + fee:.2f}"


def record_customer_interest(email: str, name: str = "not given", message: str = "") -> str:
    """Save a sales lead (only with an email the customer wrote)."""
    if not EMAIL_RE.fullmatch((email or "").strip()):
        return "Error: invalid email. Ask the customer for a valid email address."
    _append_log("leads.jsonl", {"email": email.strip(), "name": name, "message": message})
    return f"Lead saved for {name} <{email.strip()}>. The team will contact them."


def record_feedback(question: str) -> str:
    """Log feedback, a complaint, or a question the agent cannot answer."""
    _append_log("feedback.jsonl", {"question": question})
    return "Logged for the Garliclo team; they will follow up."


TOOLS = {
    "search_business_info": {
        "func": search_business_info, "params": ["query"],
        "description": "Search Garliclo's profile (products, storage, packaging, payment, team, policies). "
                       "Returns the most relevant passages."},
    "get_delivery_fee": {
        "func": get_delivery_fee, "params": ["area"],
        "description": "Delivery zone and fee for a town or area in Lebanon."},
    "calculate_quote": {
        "func": calculate_quote, "params": ["product_id", "kg", "area"],
        "description": "Exact order total (price x kg + delivery, with the 50 kg free-delivery rule). Products: "
                       + ", ".join(f"{i} = {p['name']} ${p['price']:.2f}/kg" for i, p in PRODUCTS.items()) + "."},
    "record_customer_interest": {
        "func": record_customer_interest, "params": ["email", "name", "message"],
        "description": "Save a sales lead. Use only with an email address the customer actually wrote."},
    "record_feedback": {
        "func": record_feedback, "params": ["question"],
        "description": "Log feedback, a complaint, or a question the business information does not answer."},
}

print(calculate_quote(1, 30, "Tripoli"))
print(calculate_quote(4, 50, "Dekwaneh"))
print(get_delivery_fee("Zahle"))
print(search_business_info("frozen garlic shelf life temperature")[:200], "…")

# %% [markdown]
# ## 4. Instructions block
# The instructions are written separately from the persona, following the guidelines from class:
# * **one purpose per instruction** (each rule below does exactly one thing);
# * **explicit inputs and outputs**, including the exact output format the parser expects;
# * a **worked example** inside the instructions themselves;
# * **no conflict with any persona**: personas only set tone, vocabulary, response style and values; the
#   instructions set behaviour and format, so every persona can follow them unchanged.
#
# The `{format}` and `{examples}` slots let the experiments switch Chain-of-Thought and few-shot examples on and off.

# %%
INSTRUCTIONS_TEMPLATE = """\
## Inputs you receive
- The customer's latest message (English, Arabic or French) and the recent conversation.
- After each Action, an "Observation:" message with the tool's result.

## Output format
{format}
- Never write "Observation:" yourself; the system adds it after running your Action.
- Call exactly one tool per Action, with its parameters as a JSON object.

## Rules
1. Look up facts with search_business_info before stating any policy, storage, packaging or payment detail.
2. Use calculate_quote for every order total; never do price arithmetic yourself.
3. Use get_delivery_fee when the customer asks about delivery to an area.
4. If the observations do not contain the answer, call record_feedback with the customer's question.
5. If the customer asks for something you cannot do yourself (cancel or change an order, a refund, a guaranteed
   delivery time), call record_feedback with the request and say the team will follow up.
6. Log every complaint, compliment or suggestion with record_feedback.
7. As soon as an interested customer writes an email address, call record_customer_interest.
8. Call record_customer_interest only with an email the customer actually wrote.
9. When the customer wants to order but has not given an email, ask for their name and email before you finish.
10. If "trimmed" (مقمع) or "untrimmed" (غير مقمع) could mean two products, give both options with prices.
11. Garliclo sells to businesses only; politely decline household orders.
12. Write the Answer in the language of the customer's latest message, whatever language the examples use.
13. Keep the Answer under 120 words.
{examples}"""

FORMAT_COT = """\
Each turn, write one short "Thought:" line with your reasoning, then exactly one of:
  Action: <tool_name>(<JSON object of parameters>)
  Answer: <your final reply to the customer>"""

FORMAT_DIRECT = """\
Each turn, write exactly one line and no reasoning:
  Action: <tool_name>(<JSON object of parameters>)
  Answer: <your final reply to the customer>"""

EXAMPLES = [  # the first one is the worked example; the others are extra shots for few-shot prompting
    {"customer": "How much for 30 kg of fresh trimmed garlic delivered to Tripoli?",
     "steps": [("Thought", "Fresh trimmed is product 1. I need an exact quote with delivery."),
               ("Action", 'calculate_quote({"product_id": 1, "kg": 30, "area": "Tripoli"})'),
               ("Observation", "Product 1 (Peeled & trimmed, large cloves): 30 kg x $4.00 = $120.00 | "
                               "Delivery to Tripoli: Zone 4, $10 | Total: $130.00"),
               ("Thought", "I have the total. I should offer to pass the order on."),
               ("Answer", "30 kg of fresh trimmed garlic is $120, plus $10 delivery to Tripoli (Zone 4): $130 in "
                          "total. Shall I pass your order to the team? I just need your name and email.")]},
    {"customer": "بدي 10 كيلو مقشر مقمع، أنا سامي sami@grill.lb",
     "steps": [("Thought", "He wants 10 kg of trimmed garlic and gave his email, so I save the lead."),
               ("Action", 'record_customer_interest({"email": "sami@grill.lb", "name": "Sami", '
                          '"message": "10 kg trimmed garlic"})'),
               ("Observation", "Lead saved for Sami <sami@grill.lb>. The team will contact them."),
               ("Answer", "تمام سامي! سجّلت طلبك لـ 10 كيلو ثوم مقشّر مقمّع، والفريق رح يتواصل معك قريباً. "
                          "بتحب طازة (4$ للكيلو) أو مجلّدة (3$ للكيلو)؟")]},
    {"customer": "Do you deliver on Sundays?",
     "steps": [("Thought", "Delivery days may be in the profile. I will search."),
               ("Action", 'search_business_info({"query": "delivery days Sunday"})'),
               ("Observation", "Delivery day and timing are confirmed with the team when the order is placed."),
               ("Thought", "Sundays are not covered, so I log the question for the team."),
               ("Action", 'record_feedback({"question": "Do you deliver on Sundays?"})'),
               ("Observation", "Logged for the Garliclo team; they will follow up."),
               ("Answer", "Delivery days are set with the team for each order, so I've passed your Sunday "
                          "question on and they'll confirm with you.")]},
]


def render_example(example: dict, cot: bool) -> str:
    lines = [f"Customer: {example['customer']}"]
    lines += [f"{label}: {text}" for label, text in example["steps"] if cot or label != "Thought"]
    return "\n".join(lines)


def render_instructions(cot: bool = True, shots: str = "few") -> str:
    if shots == "zero":
        examples = ""
    else:
        chosen = EXAMPLES if shots == "few" else EXAMPLES[:1]
        examples = "\n## Worked examples\n" + "\n\n".join(render_example(e, cot) for e in chosen)
    return INSTRUCTIONS_TEMPLATE.format(format=FORMAT_COT if cot else FORMAT_DIRECT, examples=examples)


print(render_instructions())

# %% [markdown]
# ## 5. Personas: prompt traits + matching LLM configuration
# Each persona has a different **task**, and is defined twice: once in words (tone, vocabulary, response style, values)
# and once in numbers (the LLM configuration). Every configuration value is chosen to support the persona's traits.
#
# | | 😊 Friendly Advisor ("Toumi") | 🧑‍🍳 Strict Expert ("Chef Nader") | 🛡️ Cautious Helper ("Rima") |
# |---|---|---|---|
# | **Task** | First contact and sales: explain products, turn interest into leads | Exact quotes, quantities and storage plans for chefs and buyers | After-sales support: complaints, problems, unanswered questions |
# | **Tone** | Warm Lebanese hospitality, light humour | Direct, professional, no small talk | Calm, empathetic, reassuring |
# | **Vocabulary** | Simple, everyday words | Technical: kg, zone, shelf life, unit price | Plain and careful; no jargon |
# | **Style** | Short paragraphs, ends with a next step | Bullet points and numbers | Acknowledge → confirm details → next step |
# | **Values** | Helpfulness, honesty, never pushy | Accuracy, precision, efficiency | Trust, safety, never over-promise |
# | **temperature** | 0.7 — natural, varied phrasing | 0.1 — near-deterministic, repeatable numbers | 0.3 — steady and consistent |
# | **top_p** | 0.9 — keeps variety, trims odd tokens | 0.5 — only the most likely tokens | 0.8 — some flexibility in wording |
# | **frequency_penalty** | 0.3 — avoids repeating stock phrases | 0.0 — repeating exact terms is fine | 0.1 — mild |
# | **presence_penalty** | 0.3 — nudges toward suggesting next steps | 0.0 — stays on the exact question | 0.0 — no drifting to new topics |
# | **max_tokens** | 350 — room for a friendly reply | 300 — enough for figures and bullets | 300 |
# | **model** | gpt-4.1-mini | gpt-4.1-mini | gpt-4.1-mini |
#
# All three personas use the same model on purpose: holding the model constant isolates the effect of prompt traits and
# sampling settings. Model type is varied separately in the configuration sweep (§10, group C).
# `top_k` is not offered by the OpenAI API; it is tested on Gemini in §12. The persona text never contradicts the
# instructions: no persona asks for long answers, invented facts, or skipping tools.

# %%
PERSONAS = {
    "friendly_advisor": {
        "label": "😊 Friendly Advisor (Toumi)",
        "task": "sales",
        "prompt": """\
You are Toumi, Garliclo's friendly sales advisor.
- Task: welcome new business customers, explain Garliclo's air-peeled garlic, and turn interest into orders.
- Tone: warm Lebanese hospitality, upbeat, with a light garlic pun now and then.
- Vocabulary: simple, everyday words; no jargon.
- Response style: short paragraphs; at most one emoji; always end with a helpful next step or question.
- Values: genuinely helpful and honest; enthusiastic but never pushy.""",
        "config": {"model": "gpt-4.1-mini", "temperature": 0.7, "top_p": 0.9,
                   "frequency_penalty": 0.3, "presence_penalty": 0.3, "max_tokens": 350},
        "why": {"temperature": "natural, varied phrasing", "top_p": "keeps variety, trims odd tokens",
                "frequency_penalty": "avoids repeating stock phrases",
                "presence_penalty": "nudges toward suggesting next steps", "max_tokens": "room for a warm reply"},
    },
    "strict_expert": {
        "label": "🧑‍🍳 Strict Expert (Chef Nader)",
        "task": "quote",
        "prompt": """\
You are Chef Nader, Garliclo's product and kitchen-planning expert.
- Task: give chefs and purchasing managers exact quotes, quantities, and storage plans.
- Tone: direct, professional, no small talk, no emojis.
- Vocabulary: precise and technical (kg, unit price, delivery zone, shelf life, storage temperature).
- Response style: bullet points with figures; state the number first, then the brief justification.
- Values: accuracy above everything; if a fact is not confirmed, say so plainly.""",
        "config": {"model": "gpt-4.1-mini", "temperature": 0.1, "top_p": 0.5,
                   "frequency_penalty": 0.0, "presence_penalty": 0.0, "max_tokens": 300},
        "why": {"temperature": "near-deterministic, repeatable numbers", "top_p": "only the most likely tokens",
                "frequency_penalty": "repeating exact terms is fine", "presence_penalty": "stays on the exact question",
                "max_tokens": "enough for figures and bullets"},
    },
    "cautious_helper": {
        "label": "🛡️ Cautious Helper (Rima)",
        "task": "support",
        "prompt": """\
You are Rima, Garliclo's careful customer-support helper.
- Task: handle complaints, order problems, and questions the business information may not answer.
- Tone: calm, empathetic, and reassuring.
- Vocabulary: plain, careful wording; no jargon.
- Response style: acknowledge the customer's concern, confirm the details you have, then state the next step.
- Values: trust and safety; never promise refunds, replacements, or delivery dates; escalate to the team.""",
        "config": {"model": "gpt-4.1-mini", "temperature": 0.3, "top_p": 0.8,
                   "frequency_penalty": 0.1, "presence_penalty": 0.0, "max_tokens": 300},
        "why": {"temperature": "steady, consistent wording", "top_p": "some flexibility in wording",
                "frequency_penalty": "mild variety", "presence_penalty": "no drifting to new topics",
                "max_tokens": "room to acknowledge and explain next steps"},
    },
}

for key, p in PERSONAS.items():
    print(f"{p['label']:32} task={p['task']:8} config={p['config']}")

# %% [markdown]
# ## 6. LLM layer: one function, every configuration knob
# `call_llm` sends a chat to either provider with the persona's configuration:
# * **OpenAI** (Chat Completions): `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `max_tokens`,
#   `stop`. OpenAI has no `top_k`, so it is reported as ignored.
# * **Gemini** (native REST API): `temperature`, `top_p`, **`top_k`**, `max_tokens`, `stop`, and `thinking_budget`
#   (Gemini 2.5's hidden reasoning, which can be switched off).
#
# The stop sequence `Observation:` keeps the model from inventing tool results: generation halts right after an Action.

# %%
PRICES = {  # USD per 1M tokens (input, output)
    "gpt-4.1": (2.00, 8.00), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40), "gpt-4o-mini": (0.15, 0.60),
}


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))  # Gemini free tier = 0
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


def call_llm(messages: list[dict], config: dict, stop: list[str] | None = None) -> dict:
    """One chat completion with the given configuration. Returns text, token counts and latency."""
    started = time.perf_counter()
    model = config["model"]
    if model.startswith("gemini"):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in messages if m["role"] != "system"]
        generation = {"temperature": config.get("temperature", 1.0), "topP": config.get("top_p", 0.95),
                      "maxOutputTokens": config.get("max_tokens", 400)}
        if config.get("top_k"):
            generation["topK"] = config["top_k"]
        if "thinking_budget" in config:  # Gemini 2.5: 0 turns hidden reasoning off
            generation["thinkingConfig"] = {"thinkingBudget": config["thinking_budget"]}
        if stop:
            generation["stopSequences"] = stop
        body = {"contents": contents, "generationConfig": generation}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                       headers={"x-goog-api-key": GEMINI_API_KEY}, json=body, timeout=60)
        r.raise_for_status()
        data = r.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        usage = data.get("usageMetadata", {})
        text = "".join(p.get("text", "") for p in parts)
        tokens_in, tokens_out = usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)
    else:
        response = openai_client.chat.completions.create(
            model=model, messages=messages, stop=stop,
            temperature=config.get("temperature", 1.0), top_p=config.get("top_p", 1.0),
            frequency_penalty=config.get("frequency_penalty", 0.0),
            presence_penalty=config.get("presence_penalty", 0.0),
            max_completion_tokens=config.get("max_tokens", 400))
        text = response.choices[0].message.content or ""
        tokens_in, tokens_out = response.usage.prompt_tokens, response.usage.completion_tokens
    return {"text": text, "tokens_in": tokens_in, "tokens_out": tokens_out,
            "ms": round((time.perf_counter() - started) * 1000),
            "ignored": ["top_k"] if config.get("top_k") and not model.startswith("gemini") else []}


print(call_llm([{"role": "user", "content": "Reply with the single word: ready"}],
               {"model": "gpt-4.1-nano", "temperature": 0, "max_tokens": 5})["text"])

# %% [markdown]
# ## 7. ReAct prompt builder
# The system prompt combines the **persona**, the **instructions block**, and the **tool descriptions**. Four switches
# create the prompt-engineering variants used in the experiments:
#
# | Switch | Default | Alternative |
# |---|---|---|
# | `cot` | `True`: a "Thought:" line before each Action/Answer | `False`: Action/Answer only, no visible reasoning |
# | `shots` | `"few"`: worked example + 2 extra examples | `"zero"`: format description only |
# | `placement` | `"system"`: everything in the system message | `"in_message"`: instructions sent inside the user message |
# | `instructions` | `True`: full instructions block | `False`: persona + tools + output format only |

# %%
DEFAULT_VARIANT = {"cot": True, "shots": "few", "placement": "system", "instructions": True}


def tools_block() -> str:
    return "\n".join(f"- {name}({', '.join(t['params'])}): {t['description']}" for name, t in TOOLS.items())


def build_prompt(persona_key: str, variant: dict) -> str:
    v = {**DEFAULT_VARIANT, **variant}
    persona = PERSONAS[persona_key]["prompt"]
    if v["instructions"]:
        guidance = "# INSTRUCTIONS\n" + render_instructions(v["cot"], v["shots"])
    else:  # experiment: only what the parser needs to work
        guidance = "# OUTPUT FORMAT\n" + (FORMAT_COT if v["cot"] else FORMAT_DIRECT)
    return (f"# PERSONA\n{persona}\n\n{guidance}\n\n# TOOLS\n{tools_block()}\n\n"
            f"Today is {datetime.date.today():%A, %d %B %Y}.")


def initial_messages(question: str, history: list[dict], persona_key: str, variant: dict) -> list[dict]:
    v = {**DEFAULT_VARIANT, **variant}
    prompt = build_prompt(persona_key, v)
    if v["placement"] == "system":
        return [{"role": "system", "content": prompt}, *history[-10:], {"role": "user", "content": question}]
    return [{"role": "system", "content": "You are an assistant. Follow the instructions in the user's message."},
            *history[-10:], {"role": "user", "content": f"{prompt}\n\n# CUSTOMER MESSAGE\n{question}"}]


print(build_prompt("strict_expert", {"shots": "zero"})[:1200], "…")

# %% [markdown]
# ## 8. The ReAct loop in LangGraph
# The loop is a small state machine, written by hand:
#
# ```
# START → think ──(Action found)──→ act ──→ think …
#           ├────(Answer found)──→ finish → END
#           └────(bad format)────→ think (with a format reminder as the observation)
# ```
#
# * **`think`** calls the LLM with the persona's configuration and the stop sequence `Observation:`, then **parses**
#   the reply into Thought / Action / Answer.
# * **Action detection** (`parse_reply`): a regex finds `Action: tool_name(...)`; arguments are read as a JSON object,
#   or as positional values mapped to the tool's parameters. An Action takes priority over an Answer in the same reply.
# * **`act`** runs the tool and stores its result as the **Observation**, which is sent back to the model as the next
#   user message.
# * **`finish`** extracts the final Answer (or a safe fallback if the step limit is reached).

# %%
ACTION_RE = re.compile(r"Action\s*:\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", re.M)
ANSWER_RE = re.compile(r"Answer\s*:\s*(.*)", re.S)
THOUGHT_RE = re.compile(r"Thought\s*:\s*(.*?)(?=\n\s*(?:Action|Answer)\s*:|\Z)", re.S)


def parse_arguments(tool: str, raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("{"):
        return json.loads(raw)
    values = ast.literal_eval(f"[{raw}]") if raw else []
    return dict(zip(TOOLS[tool]["params"], values))


def parse_reply(text: str) -> dict:
    """Detect Thought, Action (tool + arguments) and Answer in one model reply."""
    thought = THOUGHT_RE.search(text)
    action = ACTION_RE.search(text)
    answer = ANSWER_RE.search(text)
    parsed = {"thought": thought.group(1).strip() if thought else "", "action": None, "answer": None, "error": None}
    if action and (not answer or action.start() < answer.start()):
        tool = action.group(1)
        if tool not in TOOLS:
            parsed["error"] = f"Unknown tool '{tool}'. Available tools: {', '.join(TOOLS)}."
        else:
            try:
                parsed["action"] = {"tool": tool, "args": parse_arguments(tool, action.group(2))}
            except (ValueError, SyntaxError) as exc:
                parsed["error"] = f"Could not read the arguments of {tool}: {exc}. Use a JSON object."
    elif answer:
        parsed["answer"] = answer.group(1).strip()
    else:
        parsed["error"] = "Format error: reply with 'Action: tool_name({...})' or 'Answer: ...'."
    return parsed


class ReActState(TypedDict):
    question: str
    persona: str
    variant: dict
    config: dict
    messages: list[dict]   # the chat sent to the model: prompt + history + this turn's steps
    steps: list[dict]      # structured trace: thought / action / observation / answer per step
    answer: str | None
    llm_calls: int
    format_errors: int
    tokens_in: int
    tokens_out: int
    verbose: bool


def think(state: ReActState) -> dict:
    """Reasoning step: one LLM call, then parse the reply."""
    result = call_llm(state["messages"], state["config"], stop=["Observation:"])
    text = result["text"].strip()
    parsed = parse_reply(text)
    step = {**parsed, "raw": text, "observation": None, "ms": result["ms"]}
    messages = state["messages"] + [{"role": "assistant", "content": text or "(empty reply)"}]
    if parsed["error"]:  # tell the model what went wrong; the next think step retries
        step["observation"] = parsed["error"]
        messages.append({"role": "user", "content": f"Observation: {parsed['error']}"})
    if state["verbose"]:
        if parsed["thought"]:
            print(f"💭 Thought: {parsed['thought']}")
        if parsed["error"]:
            print(f"⚠️ {parsed['error']}")
    return {"messages": messages, "steps": state["steps"] + [step], "llm_calls": state["llm_calls"] + 1,
            "format_errors": state["format_errors"] + (1 if parsed["error"] else 0),
            "tokens_in": state["tokens_in"] + result["tokens_in"],
            "tokens_out": state["tokens_out"] + result["tokens_out"]}


def act(state: ReActState) -> dict:
    """Action step: run the tool and return its result as the observation."""
    step = dict(state["steps"][-1])
    tool, args = step["action"]["tool"], step["action"]["args"]
    try:
        observation = str(TOOLS[tool]["func"](**args))
    except TypeError as exc:
        observation = f"Error: wrong arguments for {tool} ({exc})."
    step["observation"] = observation
    if state["verbose"]:
        print(f"🛠️ Action: {tool}({json.dumps(args, ensure_ascii=False)})\n👀 Observation: {observation}")
    return {"steps": state["steps"][:-1] + [step],
            "messages": state["messages"] + [{"role": "user", "content": f"Observation: {observation}"}]}


def finish(state: ReActState) -> dict:
    """Final step: return the parsed Answer, or a safe fallback when the step limit was reached."""
    last = state["steps"][-1] if state["steps"] else {}
    answer = last.get("answer")
    if not answer:  # step limit reached, or the model never used the format: keep any plain text it wrote
        raw = last.get("raw", "")
        answer = raw if raw and not ACTION_RE.search(raw) and "Thought" not in raw else (
            "Sorry, I couldn't complete that. Please WhatsApp the Garliclo team at +961 70 546 473.")
    if state["verbose"]:
        print(f"✅ Answer: {answer}")
    return {"answer": answer}


def route_after_think(state: ReActState) -> str:
    """Conditional edge: decide where the loop goes after a reasoning step."""
    last = state["steps"][-1]
    if state["llm_calls"] >= MAX_LLM_CALLS:
        return "finish"
    if last["action"]:
        return "act"
    if last["answer"]:
        return "finish"
    return "think"  # format error: retry with the reminder


builder = StateGraph(ReActState)
builder.add_node("think", think)
builder.add_node("act", act)
builder.add_node("finish", finish)
builder.add_edge(START, "think")
builder.add_conditional_edges("think", route_after_think, {"act": "act", "think": "think", "finish": "finish"})
builder.add_edge("act", "think")
builder.add_edge("finish", END)
react_graph = builder.compile()


def run_react(question: str, persona_key: str = "friendly_advisor", history: list[dict] | None = None,
              variant: dict | None = None, config_overrides: dict | None = None, verbose: bool = True) -> dict:
    """Run one question through the ReAct graph and return the answer with its full trace and metrics."""
    variant = {**DEFAULT_VARIANT, **(variant or {})}
    config = {**PERSONAS[persona_key]["config"], **(config_overrides or {})}
    started = time.perf_counter()
    state = react_graph.invoke({
        "question": question, "persona": persona_key, "variant": variant, "config": config,
        "messages": initial_messages(question, history or [], persona_key, variant), "steps": [],
        "answer": None, "llm_calls": 0, "format_errors": 0, "tokens_in": 0, "tokens_out": 0, "verbose": verbose,
    }, {"recursion_limit": 4 * MAX_LLM_CALLS})
    tools_used = [s["action"]["tool"] for s in state["steps"] if s["action"]]
    return {"answer": state["answer"], "steps": state["steps"], "tools_used": tools_used,
            "llm_calls": state["llm_calls"], "format_errors": state["format_errors"],
            "tokens_in": state["tokens_in"], "tokens_out": state["tokens_out"],
            "cost": cost_usd(config["model"], state["tokens_in"], state["tokens_out"]),
            "ms": round((time.perf_counter() - started) * 1000), "config": config, "variant": variant}


print(react_graph.get_graph().draw_mermaid())
