"""
Chatbot Arena (battle) tab.
Users chat with two anonymous models.
"""

import os
import json
import time
import re
from pathlib import Path

import gradio as gr
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv

load_dotenv()

from fastchat.constants import (
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    SLOW_MODEL_MSG,
    BLIND_MODE_INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SURVEY_LINK,
)
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.gradio_block_arena_named import flash_buttons
from fastchat.serve.cloudflare_turnstile import (
    verify_turnstile,
    CLOUDFLARE_VERIFICATION_FAILED_MESSAGE,
)
from fastchat.serve.gradio_web_server import (
    State,
    bot_response,
    get_conv_log_filename,
    no_change_btn,
    enable_btn,
    disable_btn,
    invisible_btn,
    enable_text,
    disable_text,
    acknowledgment_md,
    get_ip,
    get_model_description_md,
)
from fastchat.serve.remote_logger import get_remote_logger
from fastchat.utils import (
    build_logger,
    moderation_filter,
)

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_sides = 2
enable_moderation = False
anony_names = ["", ""]
models = []

class OutageFileHandler(FileSystemEventHandler):

    def __init__(self, json_path):
        self.json_path = Path(json_path).resolve()
        self.outage_models = []
        self.load_config()
        
    def on_modified(self, event):
        if not event.is_directory and Path(event.src_path).resolve() == self.json_path:
            self.load_config()        
    
    def load_config(self):
        if not self.json_path.exists():
            self.outage_models = []
            return

        try:
            with open(self.json_path, 'r') as f:
                model_json = json.load(f)
            self.outage_models = model_json.get("outage_models", [])
            logger.info(f"Outage models reloaded from {self.json_path}!")
        except Exception as e:
            logger.warning(f"Error loading outage models from {self.json_path}: {e}")
            self.outage_models = []


outage_handler = OutageFileHandler("outage_models.json")
observer = Observer()
observer.schedule(outage_handler, path=".", recursive=False)
observer.start()


def set_global_vars_anony(enable_moderation_):
    global enable_moderation
    enable_moderation = enable_moderation_


def load_demo_side_by_side_anony(models_, url_params):
    global models
    models = models_

    states = [None] * num_sides
    selector_updates = [
        gr.Markdown(visible=True),
        gr.Markdown(visible=True),
    ]

    return states + selector_updates


def vote_last_response(states, vote_type, model_selectors, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "models": [x for x in model_selectors],
            "states": [x.dict() for x in states],
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    get_remote_logger().log(data)

    gr.Info(
        "🎉 Aitäh hindamast! Sinu tagasiside kujundab mudelite edetabeli - palun hääleta vastutustundlikult."
    )
    if ":" not in model_selectors[0]:
        for i in range(5):
            names = (
                "### Mudel A: " + states[0].model_name,
                "### Mudel B: " + states[1].model_name,
            )
            # yield names + ("",) + (disable_btn,) * 4
            yield names + (disable_text,) + (disable_btn,) * 5
            time.sleep(0.1)
    else:
        names = (
            "### Mudel A: " + states[0].model_name,
            "### Mudel B: " + states[1].model_name,
        )
        # yield names + ("",) + (disable_btn,) * 4
        yield names + (disable_text,) + (disable_btn,) * 5


def leftvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"leftvote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "leftvote", [model_selector0, model_selector1], request
    ):
        yield x


def rightvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"rightvote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "rightvote", [model_selector0, model_selector1], request
    ):
        yield x


def tievote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"tievote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "tievote", [model_selector0, model_selector1], request
    ):
        yield x


def bothbad_vote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"bothbad_vote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "bothbad_vote", [model_selector0, model_selector1], request
    ):
        yield x


def regenerate(state0, state1, request: gr.Request):
    logger.info(f"regenerate (anony). ip: {get_ip(request)}")
    states = [state0, state1]
    if state0.regen_support and state1.regen_support:
        for i in range(num_sides):
            states[i].conv.update_last_message(None)
        return (
            states + [x.to_gradio_chatbot() for x in states] + [""] + [disable_btn] * 6
        )
    states[0].skip_next = True
    states[1].skip_next = True
    return states + [x.to_gradio_chatbot() for x in states] + [""] + [no_change_btn] * 6


def clear_history(request: gr.Request):
    logger.info(f"clear_history (anony). ip: {get_ip(request)}")
    return (
        [None] * num_sides
        + [None] * num_sides
        + anony_names
        + [enable_text]
        + [invisible_btn] * 4
        + [disable_btn] * 2
        + [""]
        + [enable_btn]
    )


def share_click(state0, state1, model_selector0, model_selector1, request: gr.Request):
    logger.info(f"share (anony). ip: {get_ip(request)}")
    if state0 is not None and state1 is not None:
        vote_last_response(
            [state0, state1], "share", [model_selector0, model_selector1], request
        )


SAMPLING_WEIGHTS = {}

# target model sampling weights will be boosted.
BATTLE_TARGETS = {}

BATTLE_STRICT_TARGETS = {}

ANON_MODELS = []

SAMPLING_BOOST_MODELS = []

# outage models won't be sampled.
# OUTAGE_MODELS = ["DeepSeek-R1-Distill-Qwen-32B", "DeepSeek-R1-Distill-Qwen-1.5B"]


def get_sample_weight(model, outage_models, sampling_weights, sampling_boost_models=[]):
    if model in outage_models:
        return 0
    weight = sampling_weights.get(model, 1)
    if model in sampling_boost_models:
        weight *= 5
    return weight


def is_model_match_pattern(model, patterns):
    flag = False
    for pattern in patterns:
        pattern = pattern.replace("*", ".*")
        if re.match(pattern, model) is not None:
            flag = True
            break
    return flag


def get_battle_pair(
    models, battle_targets, outage_models, sampling_weights, sampling_boost_models
):
    

    if len(models) == 1:
        return models[0], models[0]

    model_weights = []
    for model in models:
        weight = get_sample_weight(
            model, outage_models, sampling_weights, sampling_boost_models
        )
        model_weights.append(weight)
    total_weight = np.sum(model_weights)
    model_weights = model_weights / total_weight
    # print(models)
    # print(model_weights)
    chosen_idx = np.random.choice(len(models), p=model_weights)
    chosen_model = models[chosen_idx]
    # for p, w in zip(models, model_weights):
    #     print(p, w)

    rival_models = []
    rival_weights = []
    for model in models:
        if model == chosen_model:
            continue
        if model in ANON_MODELS and chosen_model in ANON_MODELS:
            continue
        if chosen_model in BATTLE_STRICT_TARGETS:
            if not is_model_match_pattern(model, BATTLE_STRICT_TARGETS[chosen_model]):
                continue
        if model in BATTLE_STRICT_TARGETS:
            if not is_model_match_pattern(chosen_model, BATTLE_STRICT_TARGETS[model]):
                continue
        weight = get_sample_weight(model, outage_models, sampling_weights)
        if (
            weight != 0
            and chosen_model in battle_targets
            and model in battle_targets[chosen_model]
        ):
            # boost to 20% chance
            weight = 0.5 * total_weight / len(battle_targets[chosen_model])
        rival_models.append(model)
        rival_weights.append(weight)
    # for p, w in zip(rival_models, rival_weights):
    #     print(p, w)
    rival_weights = rival_weights / np.sum(rival_weights)
    rival_idx = np.random.choice(len(rival_models), p=rival_weights)
    rival_model = rival_models[rival_idx]

    swap = np.random.randint(2)
    logger.info(f"Chosen model: {chosen_model}. Rival model: {rival_model}")
    if swap == 0:
        return chosen_model, rival_model
    else:
        return rival_model, chosen_model


def add_text(
    state0,
    state1,
    model_selector0,
    model_selector1,
    text,
    turnstile_token,
    cf_verified,
    request: gr.Request,
):
    ip = get_ip(request)
    logger.info(f"add_text (anony). ip: {ip}. len: {len(text)}")
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]

    # Init states if necessary
    if states[0] is None:
        assert states[1] is None
        current_outage_models = outage_handler.outage_models
        logger.info(f"current_outage_models: {current_outage_models}")
        model_left, model_right = get_battle_pair(
            models,
            BATTLE_TARGETS,
            current_outage_models,
            SAMPLING_WEIGHTS,
            SAMPLING_BOOST_MODELS,
        )
        states = [
            State(model_left),
            State(model_right),
        ]

    if not cf_verified:
        logger.info(f"verifying cf turnstile. ip: {ip}")

        cf_verify_response = verify_turnstile(turnstile_token)

        if cf_verify_response.get("success"):
            cf_verified = True
        else:
            error_codes = cf_verify_response.get("error-codes", [])
            logger.info(f"cf verification failed! ip: {ip}. error_codes={error_codes}")

            gr.Warning(CLOUDFLARE_VERIFICATION_FAILED_MESSAGE, 20)

            for i in range(num_sides):
                states[i].skip_next = True  # skips generate call in bot_response
            return (
                states
                + [x.to_gradio_chatbot() for x in states]
                + [text]  # keeps user prompt
                + [
                    disable_btn,
                ]
                * 6
                + [""]
                + [cf_verified]
            )

    if len(text) <= 0:
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [""]
            + [
                no_change_btn,
            ]
            * 6
            + [""]
            + [cf_verified]
        )

    model_list = [states[i].model_name for i in range(num_sides)]
    # turn on moderation in battle mode
    all_conv_text_left = states[0].conv.get_prompt()
    all_conv_text_right = states[0].conv.get_prompt()
    all_conv_text = (
        all_conv_text_left[-1000:] + all_conv_text_right[-1000:] + "\nuser: " + text
    )
    flagged = moderation_filter(all_conv_text, model_list, do_moderation=True)
    if flagged:
        logger.info(f"violate moderation (anony). ip: {ip}. text: {text}")
        # overwrite the original text
        text = MODERATION_MSG

    conv = states[0].conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {get_ip(request)}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [CONVERSATION_LIMIT_MSG]
            + [
                no_change_btn,
            ]
            * 6
            + [""]
            + [cf_verified]
        )

    text = text[:BLIND_MODE_INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    for i in range(num_sides):
        states[i].conv.append_message(states[i].conv.roles[0], text)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False

    hint_msg = ""
    for i in range(num_sides):
        if states[i].model_name in ["gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-preview-05-06", "qwen3-235b-a22b"]:
            hint_msg = SLOW_MODEL_MSG
    return (
        states
        + [x.to_gradio_chatbot() for x in states]
        + [""]
        + [
            disable_btn,
        ]
        * 6
        + [hint_msg]
        + [cf_verified]
    )


def bot_response_multi(
    state0,
    state1,
    temperature,
    top_p,
    max_new_tokens,
    request: gr.Request,
):
    logger.info(f"bot_response_multi (anony). ip: {get_ip(request)}")

    if state0 is None or state0.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (
            state0,
            state1,
            state0.to_gradio_chatbot(),
            state1.to_gradio_chatbot(),
        ) + (no_change_btn,) * 6
        return

    states = [state0, state1]
    gen = []
    for i in range(num_sides):
        gen.append(
            bot_response(
                states[i],
                temperature,
                top_p,
                max_new_tokens,
                request,
                apply_rate_limit=False,
                use_recommended_config=True,
            )
        )

    model_tpy = []
    for i in range(num_sides):
        token_per_yield = 1
        if states[i].model_name in [
            "gemini-pro",
            "gemma-1.1-2b-it",
            "gemma-1.1-7b-it",
            "phi-3-mini-4k-instruct",
            "phi-3-mini-128k-instruct",
            "snowflake-arctic-instruct",
        ]:
            token_per_yield = 30
        elif states[i].model_name in [
            "qwen-max-0428",
            "qwen-vl-max-0809",
            "qwen1.5-110b-chat",
            "llava-v1.6-34b",
        ]:
            token_per_yield = 7
        elif states[i].model_name in [
            "qwen2.5-72b-instruct",
            "qwen2-72b-instruct",
            "qwen-plus-0828",
            "qwen-max-0919",
            "llama-3.1-405b-instruct-bf16",
        ]:
            token_per_yield = 4
        model_tpy.append(token_per_yield)

    chatbots = [None] * num_sides
    iters = 0
    while True:
        stop = True
        iters += 1
        for i in range(num_sides):
            try:
                # yield fewer times if chunk size is larger
                if model_tpy[i] == 1 or (iters % model_tpy[i] == 1 or iters < 3):
                    ret = next(gen[i])
                    states[i], chatbots[i] = ret[0], ret[1]
                stop = False
            except StopIteration:
                pass
        yield states + chatbots + [disable_btn] * 6
        if stop:
            break


def build_side_by_side_ui_anony(models):
    states = [gr.State() for _ in range(num_sides)]
    model_selectors = [None] * num_sides
    chatbots = [None] * num_sides
    cf_verified = gr.State(False)

    gr.HTML(
        """
            <div id="hero_text">
                <h2 style="display: flex; justify-content: center; align-items: center;">
                    <img width='22px' height='22px' style="margin-right: 6px; margin-top: 4px; height: 22px" src="https://i.imgur.com/06AMu9U.png"/>
                    TEHISARU BAROMEETER
                    <img width='22px' height='22px' style="margin-left: 6px; margin-top: 4px; height: 22px" src="https://i.imgur.com/06AMu9U.png"/>
                </h2>
                <ol>
                    <li>Küsi eesti keeles</li>
                    <li>Vali parim vastus</li>
                    <li>Tutvu edetabeliga!</li>
                </ol>
            </div>
            """,
        elem_id="hero_container",
    )

    with gr.Group(elem_id="share-region-anony"):
        with gr.Accordion(
            f"🔍 Kliki siia, et näha võrdluses olevaid mudeleid. 🔥Valikus uued mudelid!",
            open=False,
            elem_id="models_accordion",
        ):
            model_description_md = get_model_description_md(models)
            gr.Markdown(model_description_md, elem_id="model_description_markdown")
        with gr.Row():
            for i in range(num_sides):
                label = "Mudel A" if i == 0 else "Mudel B"
                with gr.Column():
                    chatbots[i] = gr.Chatbot(
                        label=label,
                        elem_classes=f"chatbot chatbot_{i}",
                        height=650,
                        latex_delimiters=[
                            {"left": "$", "right": "$", "display": False},
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": r"\(", "right": r"\)", "display": False},
                            {"left": r"\[", "right": r"\]", "display": True},
                        ],
                    )

        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    model_selectors[i] = gr.Markdown(
                        anony_names[i], elem_id="model_selector_md"
                    )
        with gr.Row():
            slow_warning = gr.Markdown("")

    with gr.Group(elem_id="fixed_footer"):
        gr.HTML(
            value="<div id='turnstile-container'></div>", elem_id="turnstile-container"
        )
        token = gr.Textbox(visible=False, elem_id="turnstile-token")

        with gr.Row(elem_id="selection_buttons_row"):
            leftvote_btn = gr.Button(
                value=" A on parem",
                elem_classes="voting_button",
                visible=False,
                interactive=False,
                variant="primary",
            )
            rightvote_btn = gr.Button(
                value="B on parem",
                elem_classes="voting_button",
                visible=False,
                interactive=False,
                variant="primary",
            )
            tie_btn = gr.Button(
                value="🤝  Viik",
                elem_classes="voting_button",
                visible=False,
                interactive=False,
            )
            bothbad_btn = gr.Button(
                value="👎  Mõlemad on halvad",
                elem_classes="voting_button",
                visible=False,
                interactive=False,
            )

        with gr.Row(elem_id="input_row"):
            textbox = gr.Textbox(
                show_label=False,
                autofocus=True,
                placeholder="👉 Kirjuta siia enda küsimus ja vajuta ENTER",
                elem_id="input_box",
            )
            send_btn = gr.Button(
                value="Saada", variant="primary", scale=0, elem_id="send_button"
            )

        with gr.Row() as button_row:
            clear_btn = gr.Button(
                value="🎲 Uus vestlus",
                elem_classes="control_button",
                interactive=False,
                visible=False,
            )
            info_btn = gr.Button(
                value="❓ Kuidas valida?", elem_classes="control_button", visible=False
            )
            regenerate_btn = gr.Button(
                value="🔄  Genereeri vastus uuesti",
                elem_classes="control_button hidden",
                visible=False,
                interactive=False,
            )

    with gr.Accordion("Parameetrid", open=False, visible=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=2048,
            value=2000,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    # gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    # Register listeners
    btn_list = [
        leftvote_btn,
        rightvote_btn,
        tie_btn,
        bothbad_btn,
        info_btn,
        clear_btn,
    ]
    leftvote_btn.click(
        leftvote_last_response,
        states + model_selectors,
        model_selectors
        + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, send_btn],
    )
    rightvote_btn.click(
        rightvote_last_response,
        states + model_selectors,
        model_selectors
        + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, send_btn],
    )
    tie_btn.click(
        tievote_last_response,
        states + model_selectors,
        model_selectors
        + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, send_btn],
    )
    bothbad_btn.click(
        bothbad_vote_last_response,
        states + model_selectors,
        model_selectors
        + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, send_btn],
    )
    # regenerate_btn.click(
    #     regenerate, states, states + chatbots + [textbox] + btn_list
    # ).then(
    #     bot_response_multi,
    #     states + [temperature, top_p, max_output_tokens],
    #     states + chatbots + btn_list,
    # ).then(
    #     flash_buttons, [], btn_list
    # )
    clear_btn.click(
        clear_history,
        None,
        states
        + chatbots
        + model_selectors
        + [textbox]
        + btn_list
        + [slow_warning]
        + [send_btn],
    )

    def info_click():
        gr.Info("HOW_TO_CHOOSE_TEXT_PLACEHOLDER", duration=None)

    info_js = """
function (a, b, c, d) {
    function editContent() {
        document.querySelectorAll('.toast-wrap').forEach(message => { 
            if (message.innerHTML.toString().includes('HOW_TO_CHOOSE_TEXT_PLACEHOLDER')) {
                message.style.width = Math.min(window.innerWidth - 32, 560) + 'px'
                message.querySelector('.toast-icon').style.alignSelf = "baseline"

                let title = message.querySelector('.toast-title')
                title.innerText = "Kuidas valida?"
                title.style.textTransform = "none"

                let text = message.querySelector('.toast-text')
                text.style.paddingTop = "8px"
                text.style.paddingLeft = "16px"

                text.innerHTML = `
<p>Abiks on järgmised küsimused:</p>
<ul>
  <li><span class='bold'>Kas vastus on korralikus eesti keeles?</span> Parim vastus on eestikeelne ning ei mõju masinlikult ega toortõlkeliselt.</li>
  <li><span class='bold'>Kas vastus toetub faktidele?</span> Võimalusel kontrolli väidete õigsust. Kui teema on keeruline, hinda, kas mudel põhjendab oma vastust usutavalt.</li>
  <li><span class='bold'>Kas vastus on selge ja asjakohane?</span> Mudel peaks vastama täpselt, ilma teemast kõrvale kaldumise ja ümmarguse jututa.</li>
  <li><span class='bold'>Kas vastus on neutraalne ja tasakaalukas?</span> Mudel ei tohiks esitada tugevalt kallutatud arvamusi ega eksitavat infot, eriti tundlike teemade puhul.</li>
</ul>
<p><b>Kui valik on keeruline, saad vestlust jätkata ning teha otsus pikema dialoogi põhjal!</b></p>
            `}
        });
    }

    setTimeout(editContent, 100)
    setTimeout(editContent, 200)
    setTimeout(editContent, 300)
    setTimeout(editContent, 400)
    
    return [a, b, c, d];
}
"""
    info_btn.click(info_click, js=info_js)

    textbox.submit(
        add_text,
        states + model_selectors + [textbox, token, cf_verified],
        states + chatbots + [textbox] + btn_list + [slow_warning] + [cf_verified],
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        lambda: "",
        [],
        [slow_warning],
    ).then(
        flash_buttons,
        [],
        btn_list,
    )

    send_btn.click(
        add_text,
        states + model_selectors + [textbox, token, cf_verified],
        states + chatbots + [textbox] + btn_list + [slow_warning] + [cf_verified],
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        lambda: "",
        [],
        [slow_warning],
    ).then(
        flash_buttons, [], btn_list
    )

    return states + model_selectors
