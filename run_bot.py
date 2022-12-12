#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Simple Bot to reply to Telegram messages using OpenAI
"""
This Bot uses the Updater class to handle the bot.
First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.
Usage:
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import argparse
import yaml
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import logging, os, random, sys, datetime, glob
from dateutil.relativedelta import relativedelta
import openai
import subprocess
import pickle
import shutil

CURRENT_DIR = os.path.dirname(__file__)

STORE_CONV = False
LOGS_FOLDER = 'log'
CHAT_FOLDER = 'chats'
PICS_FOLDER = 'pic'
CHAT_PKL = 'chat.pkl'

DEFAULT_TEXT_ENGINE = 'text-davinci-003'  # 'text-davinci-002' | 'text-davinci-003'
MAX_OUT_TOKENS = 250
MAX_IN_CHARS = 500
MAX_CONTEXT_MESSAGES = 6

STABLE_DIFFUSION_PATH = ""

DEFAULT_HUMAN = "Lorenzo"
DEFAULT_BOT = "Dabotni"

DEFAULT_PROMPT = """The following is a conversation between {bot} and his friend {human}. {bot} always does what {human} tells him to do."""

PROMPT_LINK = """

{human}: {__MSG__}

{bot}:"""

CURRENT_PROMPT = None
CHAT_WHITELIST = None


# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)



class SafeDict(dict):  # To not replace stuff if it's not available
    def __missing__(self, key):
        return '{' + key + '}'


def whitelist_filter(func):
    def inner(*args, **kwargs):
        
        if len(args) < 2:
            return
        update = args[0]
        context = args[1]
        
        if update is None or context is None:
            return
        
        interaction_allowed = check_interaction_allowed(whitelist=CHAT_WHITELIST, update=update)
        
        human_name = get_human_name(from_user=update.message.from_user)
        if human_name is None:
            human_name = DEFAULT_HUMAN
        
        user = update.message.from_user
        chat_id = update.effective_chat.id
        chat_name = update.effective_chat.title if hasattr(update.effective_chat, 'title') else None
        
        if not interaction_allowed:
            if STRICT_WHITE_LIST:  # If the white-list is strict, we reply with an error and return
                
                logger.warning('Chat {chat_id} ({chat_name}) - User {user_id} ({user_name}) has been strictly whitelisted. Sends ({message_id}): "{user_msg}"'.format(chat_name=chat_name, chat_id=chat_id, user_id=user.id, user_name=human_name, message_id=update.message.message_id, user_msg=update.message.text))
                context.bot.send_message(chat_id=chat_id, text='An error has occured. Please try again later.\nIf the problem persists, please <a href="https://latlmes.com/chatbot/help">contact support</a>.', disable_web_page_preview=True, parse_mode='HTML')
                return
            
            else:
                  # If the white-list is strict, we reply with the default and safe PROMPT
                  logger.warning('Chat {chat_id} ({chat_name}) - User {user_id} ({user_name}) has been shadow whitelisted.'.format(chat_name=chat_name, chat_id=chat_id, user_id=user.id, user_name=human_name))
                  
                  return func(*args, **kwargs, prompt_text=DEFAULT_PROMPT)
            
        else:
            return func(*args, **kwargs, prompt_text=CURRENT_PROMPT)  # If the user is whitelisted, we load the "CURRENT_PROMPT", a.k.a. the prompt in the file
    
    return inner


def check_interaction_allowed(whitelist, update):
    if whitelist is None:
        return True  # If there is no whitelist, we allow everything
    
    chat_id = update.message.chat_id
    user_id = update.message.from_user.id
    
    if chat_id in whitelist['CHATS']:
        return True
    
    if user_id in whitelist['USERS']:
        return True
    
    return False


def get_human_name(from_user):
    if hasattr(from_user, 'first_name') and hasattr(from_user, 'last_name'):
        if from_user['first_name'] is not None and len(from_user['first_name']) > 0:
            if from_user['last_name'] is not None and len(from_user['last_name']) > 0:
                return '{} {}'.format(from_user['first_name'].strip(), from_user['last_name'].strip())
    
    for field in ['first_name', 'last_name', 'username']:  # By order of preference
        if hasattr(from_user, field) and from_user[field] is not None and len(from_user[field]) > 0:
            return from_user[field].strip()
    
    return None  # If there is no luck


def create_chat_folder(effective_chat, reset_chat=False):
    chat_id = str(effective_chat.id)
    chat_folder = os.path.join(CURRENT_DIR, CHAT_FOLDER, chat_id)
    
    if reset_chat and os.path.isdir(chat_folder):
        shutil.rmtree(chat_folder)  # We erase all conversation/pictures/whatever
    
    if not os.path.isdir(chat_folder):
        os.makedirs(chat_folder, exist_ok=True)
        os.makedirs(os.path.join(chat_folder, PICS_FOLDER), exist_ok=True)
        pkl_file = os.path.join(chat_folder, CHAT_PKL)
        
        with open(pkl_file, "wb") as chat_file:
            pickle.dump({'metadata': effective_chat.to_dict(), 'chat': []}, chat_file, protocol=pickle.HIGHEST_PROTOCOL)
    
    return chat_folder


def save_interaction(chat_folder, user_name, msg_text):
    pkl_file = os.path.join(chat_folder, CHAT_PKL)
    with open(pkl_file, "rb") as chat_file:
        chat_data = pickle.load(chat_file)
    
    with open(pkl_file, "wb") as chat_file:
        new_data = {'user': user_name, 'msg': msg_text}
        chat_data['chat'].append(new_data)
        # TODO: Maybe limit it so we delete oldest interactions?

        pickle.dump(chat_data, chat_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_interaction(chat_folder):
    pkl_file = os.path.join(chat_folder, CHAT_PKL)
    
    with open(pkl_file, "rb") as pkl_file:
        chat_data = pickle.load(pkl_file)
    
    return chat_data


def clean_query(msg_text, max_chars=MAX_IN_CHARS):
    if msg_text.startswith('/'):
        msg_text = msg_text.split(' ', 1)[-1]  # We remove the '/AI ' (or whatever) part
        msg_text = msg_text.strip()
    
    if len(msg_text) == 0:
        raise ValueError('Empty query')
    
    msg_text = msg_text[:max_chars]  # So we don't end up poor

    # We add a trailing dot if the text does not end with one
    last_char = msg_text[-1]
    if last_char not in set(['.', '!', '?']):
        msg_text = msg_text + '.'
    
    return msg_text


def clean_answer(msg_text):
    answ = msg_text.strip()
    
    return answ


def assemble_context(chat_data, max_messages=MAX_CONTEXT_MESSAGES):
    messages_list = chat_data['chat']
    
    if len(messages_list) == 0:
        return None
    
    messages_list = messages_list[-max_messages:]  # We limit the length
    
    context = ""
    for i, message in enumerate(messages_list):
        new_msg = '{}: {}'.format(message['user'], message['msg'])
        
        if i < len(messages_list) - 1:
            new_msg = new_msg + '\n\n'
        
        context = context + new_msg
    
    return context


def assemble_prompt(prompt_text=DEFAULT_PROMPT, human_name=DEFAULT_HUMAN, bot_name=DEFAULT_BOT, context=None):
    prompt = prompt_text.format_map(SafeDict(human=human_name, bot=bot_name))
    
    if context is not None and len(context) > 0:
        prompt = prompt + '\n\n{__CONTEXT__}'.format_map(SafeDict(__CONTEXT__=context))
    
    prompt = prompt + PROMPT_LINK.format_map(SafeDict(human=human_name, bot=bot_name))

    return prompt


def assemble_openai_query(prompt, query, human_name=DEFAULT_HUMAN, bot_name=DEFAULT_BOT):
    query = prompt.format_map(SafeDict(__MSG__=query))  # TODO: I should escape the other potential braces in the conversation
    
    return query


def generate_image(prompt_text):
    # TODO: Do in a better way
    prompt_file = os.path.abspath(os.path.join(CURRENT_DIR, "pic.txt"))
    output_dir = os.path.abspath(os.path.join(CURRENT_DIR, "bot_imgs"))
    with open(prompt_file, "w") as f:
        f.write(prompt_text)
    
    # Generate img
    generator_script = os.path.abspath(os.path.join(STABLE_DIFFUSION_PATH, "scripts", "txt2img.py"))
    config_file = os.path.abspath(os.path.join(STABLE_DIFFUSION_PATH, "configs", "stable-diffusion", "v1-inference.yaml"))
    ckpt_file = os.path.abspath(os.path.join(STABLE_DIFFUSION_PATH, "models", "ldm", "stable-diffusion-v1", "model.ckpt"))
    subprocess.call(["python", generator_script,
                     "--config", config_file,
                     "--ckpt", ckpt_file,
                     "--prompt-file", prompt_file,
                     "--outdir", output_dir,
                     "--H", "640",
                     "--W", "576",
                     "--seed", "42",
                     "--ddim_steps", "50",
                     "--n_samples", "1",
                     "--n_iter", "1",
                     "--skip_grid"])
    
    # Retrieve img
    list_of_files = glob.glob(os.path.join(output_dir, 'samples', '*')) # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    
    return latest_file


def get_openai_answer(msg_text, text_engine=DEFAULT_TEXT_ENGINE, human_name=DEFAULT_HUMAN, bot_name=DEFAULT_BOT):
    start_sequence = "\n{}:".format(bot_name)
    restart_sequence = "\n{}:".format(human_name)

    response = openai.Completion.create(
      engine=text_engine,
      prompt=msg_text,
      temperature=0.9,
      max_tokens=250,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0.6,
      stop=[restart_sequence, start_sequence]
    )
    answ = response.choices[0].text

    return answ


def talk_to_openai(message, update, store_conv=False, prompt_text=DEFAULT_PROMPT, human_name=DEFAULT_HUMAN, bot_name=DEFAULT_BOT):
    user = update.message.from_user
    chat_id = update.effective_chat.id
	
    if store_conv:
        # We create (or get) file with chat_id conversation
        chat_folder = create_chat_folder(effective_chat=update.effective_chat)
    
        # Retrieve conv_context
        conv_context_data = load_interaction(chat_folder=chat_folder)
    
        # Store question in file
        save_interaction(chat_folder=chat_folder, user_name=human_name, msg_text=message)
        
        context_txt = assemble_context(chat_data=conv_context_data)
    
    else:
        context_txt = None
    
    prompt = assemble_prompt(prompt_text=prompt_text, human_name=human_name,
	                         bot_name=bot_name, context=context_txt)
	
    openai_query = assemble_openai_query(prompt=prompt, query=message)
    
    answ = get_openai_answer(openai_query, human_name=human_name, bot_name=bot_name)
    
    answ = clean_answer(answ)
    
    if store_conv:
        # Store answer in file
        save_interaction(chat_folder=chat_folder, user_name=bot_name, msg_text=answ)
	
    return answ


@whitelist_filter
def bot_pic_handler(update, context, prompt_text=None):
    if prompt_text is None:
        prompt_text = CURRENT_PROMPT
    
    human_name = get_human_name(from_user=update.message.from_user)
    
    # TODO: IMPLEMENT PROPERLY
    
    if human_name is None:
        human_name = DEFAULT_HUMAN
    
    user = update.message.from_user
    chat_id = update.effective_chat.id
    chat_name = update.effective_chat.title if hasattr(update.effective_chat, 'title') else None
    
    msg = clean_query(update.message.text)
    
    logger.info('Chat {chat_id} ({chat_name}) - User {user_id} ({user_name}) sends \\PIC ({message_id}): "{user_msg}"'.format(chat_name=chat_name, chat_id=chat_id, user_id=user.id, user_name=human_name, message_id=update.message.message_id, user_msg=msg))
    
    answ = talk_to_openai(message=msg, update=update, store_conv=STORE_CONV, prompt_text=prompt_text, human_name=human_name, bot_name=DEFAULT_BOT)
	
    context.bot.send_message(chat_id=update.effective_chat.id, text="Let me think...")
    
    # We clean the answer
    answ = answ.replace("{}:".format(human_name), "")
    answ = answ.replace("{}:".format(bot_name), "")
    answ = answ.replace("\n", " ")
    answ = answ.strip()
    
    latest_file = generate_image(prompt_text=answ)
    
    logger.info('Chat {chat_id} ({chat_name}) - BOT ({bot_name}) answers \\PIC ({message_id}): "{bot_answ}"'.format(chat_name=chat_name, chat_id=chat_id, bot_name=DEFAULT_BOT, message_id=update.message.message_id, bot_answ=answ))
    
    context.bot.send_photo(chat_id=update.effective_chat.id,
                           photo=open(latest_file, 'rb'),
                           caption=original_answ,
                           reply_to_message_id=update.message.message_id,
                           allow_sending_without_reply=True)


@whitelist_filter
def bot_ai_handler(update, context, prompt_text=None):
    if prompt_text is None:
        prompt_text = CURRENT_PROMPT
    
    human_name = get_human_name(from_user=update.message.from_user)
    
    if human_name is None:
        human_name = DEFAULT_HUMAN
    
    user = update.message.from_user
    chat_id = update.effective_chat.id
    chat_name = update.effective_chat.title if hasattr(update.effective_chat, 'title') else None
    
    msg = clean_query(update.message.text)
    
    logger.info('Chat {chat_id} ({chat_name}) - User {user_id} ({user_name}) sends \\AI ({message_id}): "{user_msg}"'.format(chat_name=chat_name, chat_id=chat_id, user_id=user.id, user_name=human_name, message_id=update.message.message_id, user_msg=msg))
    
    if len(msg) == 0:
        return
    
    answ = talk_to_openai(message=msg, update=update, store_conv=STORE_CONV, prompt_text=prompt_text, human_name=human_name, bot_name=DEFAULT_BOT)
    
    logger.info('Chat {chat_id} ({chat_name}) - BOT ({bot_name}) answers \\AI ({message_id}): "{bot_answ}"'.format(chat_name=chat_name, chat_id=chat_id, bot_name=DEFAULT_BOT, message_id=update.message.message_id, bot_answ=answ))
    
    if len(answ) == 0:
        return
	
    context.bot.send_message(chat_id=chat_id, text=answ)


def bot_TEXT_handler(update, context):
    human_name = get_human_name(from_user=update.message.from_user)
    
    if human_name is None:
        human_name = DEFAULT_HUMAN
    
    user = update.message.from_user
    chat_id = update.effective_chat.id
    chat_name = update.effective_chat.title if hasattr(update.effective_chat, 'title') else None
    
    msg = clean_query(update.message.text)
    
    logger.info('Chat {chat_id} ({chat_name}) - User {user_id} ({user_name}) sends TEXT ({message_id}): "{user_msg}"'.format(chat_name=chat_name, chat_id=chat_id, user_id=user.id, user_name=human_name, message_id=update.message.message_id, user_msg=msg))
	
    if STORE_CONV:
        # We create (or get) file with chat_id conversation
        chat_folder = create_chat_folder(effective_chat=update.effective_chat)
    
        # Store question in file
        save_interaction(chat_folder=chat_folder, user_name=human_name, msg_text=msg)


def bot_reset_handler(update, context):  # Resets the conversation memory of the chat
    human_name = get_human_name(from_user=update.message.from_user)
    
    if human_name is None:
        human_name = DEFAULT_HUMAN
    
    user = update.message.from_user
    chat_id = update.effective_chat.id
    chat_name = update.effective_chat.title if hasattr(update.effective_chat, 'title') else None
    
    logger.info('Chat {chat_id} ({chat_name}) - User {user_id} ({user_name}) sends \\RESET ({message_id}): "{user_msg}"'.format(chat_name=chat_name, chat_id=chat_id, user_id=user.id, user_name=human_name, message_id=update.message.message_id, user_msg=update.message.text))
    
    chat_folder = create_chat_folder(effective_chat=update.effective_chat, reset_chat=True)
    
    logger.info('Chat {chat_id} ({chat_name}) - Deleted chat for user {user_id} ({user_name}): "{chat_folder}"'.format(chat_name=chat_name, chat_id=chat_id, user_id=user.id, user_name=human_name, chat_folder=chat_folder))
    
    context.bot.send_message(chat_id=chat_id, text="I forgor 💀")


def bot_help_handler(update, context):
    human_name = get_human_name(from_user=update.message.from_user)
    
    if human_name is None:
        human_name = DEFAULT_HUMAN
    
    user = update.message.from_user
    chat_id = update.effective_chat.id
    chat_name = update.effective_chat.title if hasattr(update.effective_chat, 'title') else None
    
    logger.info('Chat {chat_id} ({chat_name}) - User {user_id} ({user_name}) sends \\HELP ({message_id})'.format(chat_name=chat_name, chat_id=chat_id, user_id=user.id, user_name=human_name, message_id=update.message.message_id))
    
    context.bot.send_message(chat_id=chat_id, text="Hello there! I'm {bot_name}, a conversational artificial intelligence.\n\nYou can query me by starting your messages with \"\\AI\".".format(bot_name=DEFAULT_BOT))


def bot_ERROR_handler(update, context):
    if hasattr(update, 'message') and hasattr(update.message, 'text'):
        message_text = update.message.text
    else:
        message_text = None
        
    if hasattr(update, 'message') and hasattr(update.message, 'chat_id'):
        chat_id = update.message.chat_id
    else:
        chat_id = None
    
    if hasattr(update, 'message') and hasattr(update.message, 'message_id'):
        message_id = update.message.message_id
    else:
        message_id = None
    
    if hasattr(update, 'effective_chat') and hasattr(update.effective_chat, 'title'):
        chat_name = update.effective_chat.title
    else:
        chat_name = None
    
    if hasattr(update, 'message') and hasattr(update.message, 'from_user'):
        user_id = update.message.from_user.id
        user_name = get_human_name(from_user=update.message.from_user)
    else:
        user_id = None
        user_name = None
    
    
    logger.warning('Chat {chat_id} ({chat_name}) - User {user_id} ({user_name}) with message ({message_id}) "{user_msg}" caused error "{error_txt}"'.format(chat_name=chat_name, chat_id=chat_id, user_id=user_id, user_name=user_name, message_id=message_id, user_msg=message_text, error_txt=context.error))
    raise context.error


def _read_whitelist_file(file_name):
    with open(file_name, 'r') as f:
        data = yaml.safe_load(f)
    
    if 'CHATS' not in data:
        data['CHATS'] = []
    
    if 'USERS' not in data:
        data['USERS'] = []
    
    # To ease lookups
    data['CHATS'] = set(data['CHATS'])
    data['USERS'] = set(data['USERS'])
    
    return data

def _read_prompt_file(file_name):
    with open(file_name, 'r') as f:
        prompt_text = f.read().strip()
    
    return prompt_text


def _read_key(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        key = lines[0].strip()
    
    return key

def main(prompt_file=None, store_conv=False, whitelist_file=None, strict_white_list=False):
    openai.api_key = _read_key(os.path.join('keys', 'openai'))
    telegram_key = _read_key(os.path.join('keys', 'telegram'))
    
    if prompt_file is not None:
        prompt_text = _read_prompt_file(prompt_file)
    else:
        prompt_text = DEFAULT_PROMPT
    
    global CURRENT_PROMPT
    CURRENT_PROMPT = prompt_text
    
    global STORE_CONV
    STORE_CONV = store_conv
    
    global CHAT_WHITELIST
    if whitelist_file is not None:
        CHAT_WHITELIST = _read_whitelist_file(file_name=whitelist_file)
    else:
        CHAT_WHITELIST = None
    
    global STRICT_WHITE_LIST
    STRICT_WHITE_LIST = strict_white_list
    
    
    # Create the EventHandler and pass it your bot's token.
    updater = Updater(telegram_key)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher
	
    dp.add_handler(CommandHandler("AI", bot_ai_handler))
    
    dp.add_handler(CommandHandler("PIC", bot_pic_handler))
    
    dp.add_handler(CommandHandler("RESET", bot_reset_handler))
    
    dp.add_handler(CommandHandler("HELP", bot_help_handler))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, bot_TEXT_handler))

    # log all errors
    dp.add_error_handler(bot_ERROR_handler)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tracks a target inside a video and displays the results')
    parser.add_argument('--prompt', metavar='file', default=None, help='file storing the prompt')
    parser.add_argument('--white-list', metavar='file', default=None, help='user/chat whitelist to filter interactions')
    parser.add_argument('--strict-white-list', action='store_true', help='only allows whitelisted users to run the language model')
    parser.add_argument('--store-chats', action='store_true', help='store chats for logging and answering')
    args = parser.parse_args()
    
    # We create the necessary dirs
    os.makedirs(os.path.join(CURRENT_DIR, CHAT_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(CURRENT_DIR, LOGS_FOLDER), exist_ok=True)
    
    logs_file = os.path.join(CURRENT_DIR, LOGS_FOLDER, 'log.log')
    
    fh = logging.FileHandler(logs_file)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    
    main(prompt_file=args.prompt, store_conv=args.store_chats, whitelist_file=args.white_list, strict_white_list=args.strict_white_list)

